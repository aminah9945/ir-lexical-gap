"""E1 (classical baselines), E2 (preprocessing ablation), E5 (query expansion), BM25 sensitivity.

Usage: python src/run_lexical.py [dataset ...]
Writes results/perquery_lexical_<dataset>.csv, results/boolean_sets_<dataset>.csv,
results/bm25_grid_<dataset>.csv and cache/runs/*.pkl
"""
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.corpus import wordnet as wn

from common import (DATASETS, RES, SEED, Index, bm25_matrix, tfidf_matrix, query_matrix,
                    search, to_run, evaluate_run, per_query_rows, save_run, tokenize,
                    LUCENE_STOP, _stemmer)
import os

VARIANTS = {"raw": (False, False), "stop": (False, True), "stem": (True, False), "stem+stop": (True, True)}


def run_system(idx, W, Q, qids, name, rows, save=False, **kw):
    t = time.time()
    res = search(W, Q, **kw)
    run = to_run(res, qids, idx.doc_ids)
    per = evaluate_run(run, idx.qrels)
    rows.extend(per_query_rows(per, idx.name, name))
    if save:
        save_run(run, idx.name, name)
    m = np.mean([v["ndcg@10"] for v in per.values()])
    print(f"  {name:28s} nDCG@10={m:.4f}  ({time.time()-t:.1f}s)", flush=True)
    return run


def rm3_queries(C, W_bm25, Q, res, fb_docs=10, fb_terms=10, orig_w=0.5):
    """RM3 as in Anserini: L1-normalised feedback-doc vectors weighted by doc score."""
    Cl1 = sp.diags(1 / np.maximum(np.asarray(C.sum(1)).ravel(), 1)) @ C
    Cl1 = Cl1.tocsr()
    rows, cols, vals = [], [], []
    for i, (didx, sc) in enumerate(res):
        q = Q[i].toarray().ravel()
        if q.sum() > 0:
            q = q / q.sum()
        if len(didx) == 0:
            fb = np.zeros_like(q)
        else:
            d, s = didx[:fb_docs], sc[:fb_docs]
            fbv = np.asarray((sp.diags(s) @ Cl1[d]).sum(0)).ravel()
            top = np.argpartition(-fbv, min(fb_terms, len(fbv) - 1))[:fb_terms]
            fb = np.zeros_like(fbv)
            fb[top] = fbv[top]
            if fb.sum() > 0:
                fb /= fb.sum()
        new = orig_w * q + (1 - orig_w) * fb
        nz = np.nonzero(new)[0]
        rows += [i] * len(nz)
        cols += nz.tolist()
        vals += new[nz].tolist()
    return sp.csr_matrix((vals, (rows, cols)), shape=Q.shape, dtype=np.float32)


def wordnet_queries(qtexts, vocab, syn_w=0.3, max_syn=3):
    """Add up to `max_syn` WordNet synonyms per query term (first two synsets), weight syn_w."""
    rows, cols, vals = [], [], []
    for i, t in enumerate(qtexts):
        w = defaultdict(float)
        toks = [x for x in tokenize(t) if x not in LUCENE_STOP]
        orig = set(_stemmer.stemWords(toks))
        for tok, st in zip(toks, _stemmer.stemWords(toks)):
            if st in vocab:
                w[vocab[st]] += 1.0
            if tok.isdigit() or len(tok) < 3:
                continue
            syns = []
            for ss in wn.synsets(tok)[:2]:
                for lem in ss.lemma_names():
                    lem = lem.lower()
                    if "_" in lem or lem == tok:
                        continue
                    s2 = _stemmer.stemWord(lem)
                    if s2 not in orig and s2 in vocab and s2 not in syns:
                        syns.append(s2)
            for s2 in syns[:max_syn]:
                w[vocab[s2]] += syn_w
        for j, v in w.items():
            rows.append(i)
            cols.append(j)
            vals.append(v)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(qtexts), len(vocab)), dtype=np.float32)


def main(name):
    print(f"=== {name}", flush=True)
    idx = Index(name)
    qids = list(idx.qrels)
    qtexts = [idx.queries[q] for q in qids]
    rows = []

    # ---- E2: preprocessing ablation for TF-IDF and BM25
    for vname, (stem, stop) in VARIANTS.items():
        C, vocab, an = idx.variant(stem, stop)
        Q = query_matrix(qtexts, vocab, an)
        main_cfg = vname == "stem+stop"
        Wb, _ = bm25_matrix(C)
        run_system(idx, Wb, Q, qids, "bm25" if main_cfg else f"bm25[{vname}]", rows, save=main_cfg)
        Wt, idf = tfidf_matrix(C)
        Qt = Q.copy()
        Qt.data = (1 + np.log(Qt.data)) * idf[Qt.indices]
        run_system(idx, Wt, Qt, qids, "tfidf" if main_cfg else f"tfidf[{vname}]", rows, save=main_cfg)
        if not main_cfg:  # free memory (Touche/Quora indexes are large)
            idx._variants.pop((stem, stop), None)
            del C, Wb, Wt

    C, vocab, an = idx.variant(True, True)
    Q = query_matrix(qtexts, vocab, an)
    Qbin = query_matrix(qtexts, vocab, an, binary=True)
    B = C.copy().astype(np.float32)
    B.data[:] = 1

    # ---- E1: Boolean models
    nterms = np.asarray(Qbin.sum(1)).ravel()
    run_system(idx, B, Qbin, qids, "bool_or", rows, save=True, tie_noise=0.5)
    res_and = search(B, Qbin, k=10 ** 9, tie_noise=0.5, min_score=np.maximum(nterms, 1))
    run_and = to_run([(i[:1000], s[:1000]) for i, s in res_and], qids, idx.doc_ids)
    per = evaluate_run(run_and, idx.qrels)
    rows.extend(per_query_rows(per, name, "bool_and"))
    save_run(run_and, name, "bool_and")
    print(f"  {'bool_and':28s} nDCG@10={np.mean([v['ndcg@10'] for v in per.values()]):.4f}")
    # set-based evaluation for Boolean AND (the model's native output is a set)
    srows = []
    for q, (didx, _) in zip(qids, res_and):
        retrieved = {idx.doc_ids[i] for i in didx} - {q}
        pos = {d for d, g in idx.qrels[q].items() if g > 0}
        tp = len(retrieved & pos)
        P = tp / len(retrieved) if retrieved else 0.0
        R = tp / len(pos)
        F = 2 * P * R / (P + R) if P + R > 0 else 0.0
        srows.append(dict(dataset=name, qid=q, set_size=len(retrieved), precision=P, recall=R, f1=F,
                          n_terms=int(nterms[qids.index(q)]) if False else None))
    sdf = pd.DataFrame(srows)
    sdf["n_terms"] = nterms
    sdf.to_csv(os.path.join(RES, f"boolean_sets_{name}.csv"), index=False)

    # ---- E5: query expansion
    Wb, idf = bm25_matrix(C)
    res = search(Wb, Q, k=10)
    Qrm3 = rm3_queries(C, Wb, Q, res)
    run_system(idx, Wb, Qrm3, qids, "bm25_rm3", rows, save=True)
    Qwn = wordnet_queries(qtexts, vocab)
    run_system(idx, Wb, Qwn, qids, "bm25_wordnet", rows, save=True)

    pd.DataFrame(rows).to_csv(os.path.join(RES, f"perquery_lexical_{name}.csv"), index=False)

    # ---- BM25 k1/b sensitivity (ORACLE: tuned on test; reported only as an upper bound)
    rng = np.random.default_rng(SEED)
    sub = np.arange(len(qids)) if len(qids) <= 1000 else np.sort(rng.choice(len(qids), 1000, replace=False))
    sq = [qids[i] for i in sub]
    grid = []
    for k1 in [0.6, 0.9, 1.2, 1.5, 2.0]:
        for b in [0.2, 0.4, 0.6, 0.75, 0.9]:
            Wg, _ = bm25_matrix(C, k1, b)
            r = to_run(search(Wg, Q[sub], k=100), sq, idx.doc_ids)
            per = evaluate_run(r, {q: idx.qrels[q] for q in sq})
            grid.append(dict(dataset=name, k1=k1, b=b, ndcg10=np.mean([v["ndcg@10"] for v in per.values()])))
    g = pd.DataFrame(grid)
    g.to_csv(os.path.join(RES, f"bm25_grid_{name}.csv"), index=False)
    best = g.loc[g.ndcg10.idxmax()]
    print(f"  grid best k1={best.k1} b={best.b} nDCG@10={best.ndcg10:.4f} (default 0.9/0.4: "
          f"{g[(g.k1==0.9)&(g.b==0.4)].ndcg10.values[0]:.4f})", flush=True)


if __name__ == "__main__":
    for d in (sys.argv[1:] or DATASETS):
        main(d)
