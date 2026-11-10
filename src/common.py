"""Shared utilities: data loading, tokenisation/indexing, evaluation, run I/O.

All randomness is driven by SEED. Paths are relative to the project root.
"""
import json
import os
import re
import pickle
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import Stemmer
from sklearn.feature_extraction.text import CountVectorizer

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
for _d in (CACHE, RES, FIG):
    os.makedirs(_d, exist_ok=True)

DATASETS = ["nfcorpus", "scifact", "fiqa", "scidocs", "arguana",
            "trec-covid", "quora", "webis-touche2020"]
if os.environ.get("IRGAP_DATASETS"):  # optional subset for smoke tests only; verdict rules assume all 8
    DATASETS = os.environ["IRGAP_DATASETS"].split(",")
CORE = ["nfcorpus", "scifact", "fiqa"]  # the three datasets of the original proposal
DOMAIN = {"nfcorpus": "biomedical", "scifact": "scientific", "fiqa": "financial",
          "scidocs": "scientific", "arguana": "argument", "trec-covid": "biomedical",
          "quora": "general (dup. questions)", "webis-touche2020": "argument"}
PRETTY = {"nfcorpus": "NFCorpus", "scifact": "SciFact", "fiqa": "FiQA-2018",
          "scidocs": "SciDocs", "arguana": "ArguAna", "trec-covid": "TREC-COVID",
          "quora": "Quora", "webis-touche2020": "Touche-2020"}

# Lucene's default English stopword set (what Pyserini/Anserini use for BEIR BM25).
LUCENE_STOP = set("""a an and are as at be but by for if in into is it no not of on or
such that the their then there these they this to was will with""".split())

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_stemmer = Stemmer.Stemmer("porter")


# ----------------------------------------------------------------------------- data
def load_dataset(name):
    d = os.path.join(DATA, name)
    doc_ids, texts = [], []
    with open(os.path.join(d, "corpus.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            doc_ids.append(str(o["_id"]))
            texts.append((o.get("title") or "") + " " + (o.get("text") or ""))
    queries = {}
    with open(os.path.join(d, "queries.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            queries[str(o["_id"])] = o["text"]
    qrels = defaultdict(dict)
    with open(os.path.join(d, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            q, doc, s = line.rstrip("\n").split("\t")
            qrels[q][doc] = int(s)
    # keep only queries that have at least one positive judgement (BEIR/trec_eval practice)
    qrels = {q: r for q, r in qrels.items() if any(v > 0 for v in r.values()) and q in queries}
    queries = {q: queries[q] for q in qrels}
    return doc_ids, texts, queries, qrels


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# ----------------------------------------------------------------------------- index
class Index:
    """Raw count matrix built once; stem/stop variants derived by column mapping."""

    def __init__(self, name):
        self.name = name
        path = os.path.join(CACHE, f"{name}_index.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.__dict__.update(pickle.load(f))
            return
        doc_ids, texts, queries, qrels = load_dataset(name)
        cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\w+", dtype=np.int32)
        C = cv.fit_transform(texts).tocsr()
        raw_vocab = cv.get_feature_names_out().tolist()
        self.doc_ids, self.queries, self.qrels = doc_ids, queries, qrels
        self.raw_vocab = raw_vocab
        self.C_raw = C
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=4)

    def variant(self, stem=True, stop=True):
        """Return (doc-term count matrix, vocab dict term->col, analyser fn)."""
        key = (stem, stop)
        if not hasattr(self, "_variants"):
            self._variants = {}
        if key in self._variants:
            return self._variants[key]
        raw = self.raw_vocab
        terms = _stemmer.stemWords(raw) if stem else raw
        keep = np.array([t not in LUCENE_STOP for t in raw]) if stop else np.ones(len(raw), bool)
        vocab = {}
        rows, cols = [], []
        for i, (t, k) in enumerate(zip(terms, keep)):
            if not k:
                continue
            j = vocab.setdefault(t, len(vocab))
            rows.append(i)
            cols.append(j)
        M = sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(len(raw), len(vocab)))
        C = (self.C_raw @ M).tocsr()
        C.sort_indices()

        def analyse(text, _stem=stem, _stop=stop):
            toks = tokenize(text)
            if _stop:
                toks = [t for t in toks if t not in LUCENE_STOP]
            if _stem:
                toks = _stemmer.stemWords(toks)
            return toks

        self._variants[key] = (C, vocab, analyse)
        return self._variants[key]


def query_matrix(qtexts, vocab, analyse, binary=False):
    rows, cols, vals = [], [], []
    for i, t in enumerate(qtexts):
        counts = defaultdict(int)
        for tok in analyse(t):
            j = vocab.get(tok)
            if j is not None:
                counts[j] += 1
        for j, c in counts.items():
            rows.append(i)
            cols.append(j)
            vals.append(1.0 if binary else float(c))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(qtexts), len(vocab)), dtype=np.float32)


# ----------------------------------------------------------------------------- weighting
def bm25_matrix(C, k1=0.9, b=0.4):
    """Lucene-style BM25 doc weights: idf * tf*(k1+1)/(tf + k1*(1-b+b*dl/avgdl))."""
    C = C.tocsr().astype(np.float32)
    N = C.shape[0]
    df = np.bincount(C.indices, minlength=C.shape[1])
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    dl = np.asarray(C.sum(1)).ravel()
    avgdl = dl.mean()
    W = C.copy()
    rep_dl = np.repeat(dl, np.diff(C.indptr))
    tf = W.data
    W.data = tf * (k1 + 1) / (tf + k1 * (1 - b + b * rep_dl / avgdl)) * idf[W.indices]
    return W, idf


def tfidf_matrix(C):
    """SMART ltc: (1+log tf)*log(N/df), cosine-normalised. Returns (W, idf)."""
    C = C.tocsr().astype(np.float32)
    N = C.shape[0]
    df = np.bincount(C.indices, minlength=C.shape[1])
    idf = np.log(N / np.maximum(df, 1)).astype(np.float32)
    W = C.copy()
    W.data = (1 + np.log(W.data)) * idf[W.indices]
    norms = np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel())
    norms[norms == 0] = 1
    W = sp.diags(1 / norms) @ W
    return W.tocsr(), idf


def search(W, Q, k=1000, batch=128, tie_noise=None, min_score=None):
    """Score all docs: S = Q @ W^T. Returns list of (doc_idx array, score array).

    tie_noise: if given, add uniform noise of that magnitude (seeded) to break ties
    randomly (used for Boolean models whose scores are integers).
    min_score: per-query array; docs scoring below it are not retrieved (Boolean AND).
    """
    rng = np.random.default_rng(SEED)
    WT = W.T.tocsr()
    out = []
    for s in range(0, Q.shape[0], batch):
        Qb = Q[s:s + batch]
        S = np.asarray((Qb @ WT).todense(), dtype=np.float32)
        for i in range(S.shape[0]):
            raw = S[i]
            if min_score is not None:
                valid = np.where(raw >= min_score[s + i] - 1e-6)[0]
            else:
                valid = np.where(raw > 0)[0]
            row = raw
            if tie_noise and len(valid):
                row = raw.copy()
                row[valid] += rng.uniform(0, tie_noise, len(valid)).astype(np.float32)
            kk = min(k, len(valid))
            if kk == 0:
                out.append((np.array([], int), np.array([], np.float32)))
                continue
            sub = row[valid]
            top = np.argpartition(-sub, kk - 1)[:kk]
            top = top[np.argsort(-sub[top], kind="stable")]
            out.append((valid[top], sub[top]))
    return out


def to_run(results, qids, doc_ids, drop_identical=True):
    """Convert search output to {qid: [(docid, score), ...]} (BEIR ignore_identical_ids)."""
    run = {}
    for q, (idx, sc) in zip(qids, results):
        lst = [(doc_ids[i], float(s)) for i, s in zip(idx, sc)]
        if drop_identical:
            lst = [x for x in lst if x[0] != q]
        run[q] = lst
    return run


# ----------------------------------------------------------------------------- evaluation
def evaluate_query(ranked, rels):
    """Per-query metrics. ranked: list of docids; rels: {docid: grade}. Linear gains (trec_eval)."""
    pos = {d: g for d, g in rels.items() if g > 0}
    nrel = len(pos)
    gains = [pos.get(d, 0) for d in ranked]
    # nDCG@10
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains[:10]))
    ideal = sorted(pos.values(), reverse=True)[:10]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    # MRR@10
    mrr = 0.0
    for i, g in enumerate(gains[:10]):
        if g > 0:
            mrr = 1 / (i + 1)
            break
    # Recall@100, P@10
    hits100 = sum(1 for g in gains[:100] if g > 0)
    rec = hits100 / nrel
    p10 = sum(1 for g in gains[:10] if g > 0) / 10
    # MAP@100 (trec_eval map_cut_100)
    h, ap = 0, 0.0
    for i, g in enumerate(gains[:100]):
        if g > 0:
            h += 1
            ap += h / (i + 1)
    ap /= nrel
    rec1000 = sum(1 for g in gains[:1000] if g > 0) / nrel
    return {"ndcg@10": ndcg, "map@100": ap, "recall@100": rec, "mrr@10": mrr,
            "p@10": p10, "recall@1000": rec1000}


def evaluate_run(run, qrels):
    per = {}
    for q, rels in qrels.items():
        ranked = [d for d, _ in run.get(q, [])]
        per[q] = evaluate_query(ranked, rels)
    return per


def save_run(run, dataset, system, k=1000):
    d = os.path.join(CACHE, "runs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{dataset}__{system}.pkl"), "wb") as f:
        pickle.dump({q: v[:k] for q, v in run.items()}, f, protocol=4)


def load_run(dataset, system):
    with open(os.path.join(CACHE, "runs", f"{dataset}__{system}.pkl"), "rb") as f:
        return pickle.load(f)


def per_query_rows(per, dataset, system):
    return [dict(dataset=dataset, system=system, qid=q, **m) for q, m in per.items()]
