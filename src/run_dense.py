"""E6: dense retrieval (exact cosine search) and BM25+dense hybrids.

Usage: python src/run_dense.py <model_key> [dataset ...]
model_key in {minilm, bge}. Requires run_lexical.py to have produced the bm25 run.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from common import (CACHE, DATASETS, RES, load_dataset, evaluate_run, per_query_rows,
                    save_run, load_run)

MODELS = {"minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
          "bge": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ")}
torch.set_num_threads(int(os.environ.get("THREADS", "4")))


def encode(model, texts, path, bs=int(os.environ.get("BATCH", "64"))):
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    t = time.time()
    E = model.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False,
                     convert_to_numpy=True)
    np.save(path, E.astype(np.float16))
    print(f"    encoded {len(texts)} in {time.time()-t:.0f}s", flush=True)
    return E.astype(np.float32)


def dense_search(Dm, Qm, k=1000, batch=256):
    out = []
    for s in range(0, len(Qm), batch):
        S = Qm[s:s + batch] @ Dm.T
        for row in S:
            kk = min(k, len(row))
            top = np.argpartition(-row, kk - 1)[:kk]
            top = top[np.argsort(-row[top])]
            out.append((top, row[top]))
    return out


def rrf(runs, k=60, depth=1000):
    fused = {}
    for q in runs[0]:
        sc = {}
        for r in runs:
            for rank, (d, _) in enumerate(r.get(q, [])[:depth]):
                sc[d] = sc.get(d, 0) + 1 / (k + rank + 1)
        fused[q] = sorted(sc.items(), key=lambda x: -x[1])[:depth]
    return fused


def convex(r1, r2, alpha=0.5, depth=1000):
    """alpha*minmax(r1) + (1-alpha)*minmax(r2); unseen docs get 0 from that system."""
    fused = {}
    for q in r1:
        sc = {}
        for r, w in ((r1, alpha), (r2, 1 - alpha)):
            lst = r.get(q, [])[:depth]
            if not lst:
                continue
            v = np.array([s for _, s in lst])
            lo, hi = v.min(), v.max()
            for d, s in lst:
                sc[d] = sc.get(d, 0) + w * ((s - lo) / (hi - lo) if hi > lo else 1.0)
        fused[q] = sorted(sc.items(), key=lambda x: -x[1])[:depth]
    return fused


def main(mkey, name, model):
    print(f"=== {mkey} {name}", flush=True)
    doc_ids, texts, queries, qrels = load_dataset(name)
    qids = list(qrels)
    prefix = MODELS[mkey][1] if name != "quora" else ""
    D = encode(model, texts, os.path.join(CACHE, f"emb_{mkey}_{name}_docs.npy"))
    Qm = encode(model, [prefix + queries[q] for q in qids], os.path.join(CACHE, f"emb_{mkey}_{name}_q.npy"))
    res = dense_search(D, Qm)
    run = {q: [(doc_ids[i], float(s)) for i, s in zip(idx, sc) if doc_ids[i] != q] for q, (idx, sc) in zip(qids, res)}
    save_run(run, name, mkey)
    rows = []
    per = evaluate_run(run, qrels)
    rows += per_query_rows(per, name, mkey)
    bm25 = load_run(name, "bm25")
    for sysname, r in ((f"hybrid_rrf_{mkey}", rrf([bm25, run])),
                       (f"hybrid_cc_{mkey}", convex(bm25, run))):
        save_run(r, name, sysname)
        rows += per_query_rows(evaluate_run(r, qrels), name, sysname)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"perquery_{mkey}_{name}.csv"), index=False)
    print(df.groupby("system")["ndcg@10"].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    mkey = sys.argv[1]
    model = SentenceTransformer(MODELS[mkey][0], device=os.environ.get("DEVICE", "cpu"))
    model.max_seq_length = int(os.environ.get("MAXLEN", "256"))
    for d in (sys.argv[2:] or DATASETS):
        main(mkey, d, model)
