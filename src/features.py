"""E3: per-query vocabulary-overlap features and per-domain corpus statistics.

Primary (pre-specified) overlap measure: `cov_mean` = fraction of the query's distinct
(stemmed, stopped) terms that occur in a relevant document, averaged over the query's
relevant documents. Others are robustness variants.
Writes results/features_<dataset>.csv and results/domain_stats.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from common import DATASETS, RES, SEED, Index, LUCENE_STOP, bm25_matrix, tokenize, load_run

RARE_ZIPF = 3.0  # words rarer than ~1 per million tokens in general English are "jargon"
_zcache = {}


def zipf(w):
    if w not in _zcache:
        _zcache[w] = zipf_frequency(w, "en")
    return _zcache[w]


def jargon_rate(tokens):
    toks = [t for t in tokens if t not in LUCENE_STOP and not t.isdigit()]
    if not toks:
        return np.nan
    return float(np.mean([zipf(t) < RARE_ZIPF for t in toks]))


def main(name):
    idx = Index(name)
    C, vocab, an = idx.variant(True, True)
    _, idf = bm25_matrix(C)
    did = {d: i for i, d in enumerate(idx.doc_ids)}
    B = C.tocsr()
    bm25 = load_run(name, "bm25")
    rows, skipped = [], []
    for q, rels in idx.qrels.items():
        toks = an(idx.queries[q])
        qterms = sorted({vocab[t] for t in toks if t in vocab})
        n_oov = len({t for t in toks if t not in vocab})
        qset = set(qterms)
        pos = [d for d, g in rels.items() if g > 0 and d in did]
        if not pos:  # relevant docs missing from the corpus: overlap undefined
            skipped.append(q)
            continue
        covs, icovs, jacs, dls = [], [], [], []
        qidf = idf[qterms].sum() if qterms else 0
        for d in pos:
            row = B[did[d]]
            dset = set(row.indices.tolist())
            inter = qset & dset
            covs.append(len(inter) / len(qset) if qset else 0)
            icovs.append(idf[list(inter)].sum() / qidf if qidf > 0 else 0)
            jacs.append(len(inter) / len(qset | dset) if qset | dset else 0)
            dls.append(row.sum())
        # ambiguity proxy: coverage of the top-10 BM25 *non-relevant* documents
        nonrel = [d for d, _ in bm25.get(q, [])[:10] if rels.get(d, 0) <= 0][:10]
        nr_cov = [len(qset & set(B[did[d]].indices.tolist())) / len(qset) for d in nonrel if qset]
        raw_toks = tokenize(idx.queries[q])
        rows.append(dict(
            dataset=name, qid=q, q_len=len(raw_toks), q_terms=len(qset),
            q_oov_frac=n_oov / max(1, n_oov + len(qset)),
            n_rel=len(pos), cov_mean=np.mean(covs), cov_max=np.max(covs),
            idfcov_mean=np.mean(icovs), idfcov_max=np.max(icovs), jac_mean=np.mean(jacs),
            rel_doc_len=np.mean(dls), q_mean_idf=float(idf[qterms].mean()) if qterms else 0,
            q_jargon=jargon_rate(raw_toks),
            nonrel_top10_cov=np.mean(nr_cov) if nr_cov else np.nan))
    df = pd.DataFrame(rows)
    print(f"{name}: {len(df)} queries with features, {len(skipped)} skipped (relevant docs not in corpus)", flush=True)
    df.to_csv(os.path.join(RES, f"features_{name}.csv"), index=False)

    # domain-level stats
    rng = np.random.default_rng(SEED)
    from common import load_dataset
    _, texts, _, _ = load_dataset(name)
    samp = rng.choice(len(texts), min(2000, len(texts)), replace=False)
    corpus_jargon = np.nanmean([jargon_rate(tokenize(texts[i])[:300]) for i in samp])
    dl = np.asarray(C.sum(1)).ravel()
    return dict(dataset=name, n_docs=C.shape[0], n_queries=len(df), vocab=C.shape[1],
                avg_doc_len=float(np.mean([len(tokenize(texts[i])) for i in samp])),
                avg_doc_terms_indexed=float(dl.mean()),
                avg_q_len=df.q_len.mean(), avg_rel=df.n_rel.mean(),
                q_jargon=df.q_jargon.mean(), corpus_jargon=corpus_jargon,
                q_mean_idf=df.q_mean_idf.mean(), cov_mean=df.cov_mean.mean(),
                idfcov_mean=df.idfcov_mean.mean(), q_oov=df.q_oov_frac.mean(),
                n_skipped=len(skipped))


if __name__ == "__main__":
    out = [main(d) for d in (sys.argv[1:] or DATASETS)]
    pd.DataFrame(out).to_csv(os.path.join(RES, "domain_stats.csv"), index=False)
    print(pd.DataFrame(out).round(3).to_string())
