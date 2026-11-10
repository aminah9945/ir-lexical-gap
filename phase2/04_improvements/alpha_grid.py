"""E1/E2 input: per-query metrics of the convex hybrid H_alpha = alpha*minmax(B) + (1-alpha)*minmax(D)
for alpha in {0, 0.1, ..., 1}, plus the fused top-10 doc ids per alpha (the E2 cross-encoder pool).

Fusion is numerically identical to common2.convex / src/run_dense.py::convex (same min-max, same insertion order,
stable sort), checked against the Phase-1 hybrid_cc_<D> per-query nDCG@10 at alpha=0.5.
Datasets and dense models are discovered from the files present (Phase-1 pickles or Colab npz).

Usage: python alpha_grid.py [model ...]       (default: every dense model with runs)
Outputs: results/alpha_grid_<model>_<ds>.csv.gz (qid, alpha, ndcg@10, recall@100, mrr@10),
         cache/fused_top10_<model>_<ds>.pkl ({alpha: {qid: [docid]*10}})
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03_experiments"))
import common2 as c2  # noqa: E402

RES4 = os.path.join(HERE, "results")
CACHE4 = os.path.join(HERE, "cache")
os.makedirs(RES4, exist_ok=True)
os.makedirs(CACHE4, exist_ok=True)
ALPHAS = np.round(np.linspace(0, 1, 11), 1)
DEPTH = 1000


def read_qrels(ds):
    from pooling_bias import read_qrels as rq
    return rq(ds)


def fuse_all(lb, ld, alphas=ALPHAS, depth=DEPTH):
    """Return {alpha: ranked docid list} for one query (mirrors common2.convex)."""
    lb, ld = lb[:depth], ld[:depth]
    order, idx = [], {}
    nb = np.zeros(0)
    parts = []
    for lst in (lb, ld):
        v = np.array([s for _, s in lst], float)
        if len(v):
            lo, hi = v.min(), v.max()
            v = (v - lo) / (hi - lo) if hi > lo else np.ones_like(v)
        parts.append(v)
    for d, _ in lb:
        if d not in idx:
            idx[d] = len(order)
            order.append(d)
    for d, _ in ld:
        if d not in idx:
            idx[d] = len(order)
            order.append(d)
    n = len(order)
    sb, sd = np.zeros(n), np.zeros(n)
    for (d, _), v in zip(lb, parts[0]):
        sb[idx[d]] += v
    for (d, _), v in zip(ld, parts[1]):
        sd[idx[d]] += v
    order = np.array(order, dtype=object)
    out = {}
    for a in alphas:
        # convex() adds w*norm only for lists the doc appears in; absent -> 0 contribution (same here)
        s = a * sb + (1 - a) * sd
        o = np.argsort(-s, kind="stable")[:depth]
        out[float(a)] = list(order[o])
    return out


def run(model):
    dss = [d for d in c2.ALL_DATASETS if c2.has_run(d, "bm25") and c2.has_run(d, model)]
    for ds in dss:
        fo = os.path.join(RES4, f"alpha_grid_{model}_{ds}.csv.gz")
        if os.path.exists(fo):
            print("skip", fo)
            continue
        t0 = time.time()
        qrels = read_qrels(ds)
        B = c2.load_run(ds, "bm25")
        D = c2.load_run(ds, model)
        rows, top10 = [], {float(a): {} for a in ALPHAS}
        for q, rels in qrels.items():
            fused = fuse_all(B.get(q, []), D.get(q, []))
            for a, lst in fused.items():
                m = c2.common.evaluate_query(lst, rels)
                rows.append((q, a, m["ndcg@10"], m["recall@100"], m["mrr@10"]))
                top10[a][q] = lst[:10]
        df = pd.DataFrame(rows, columns=["qid", "alpha", "ndcg@10", "recall@100", "mrr@10"])
        # check alpha=0.5 against the existing hybrid labels
        pq = c2.perquery(ds)
        hs = f"hybrid_cc_{model}"
        if hs in pq.columns:
            h = df[df.alpha == 0.5].set_index("qid")["ndcg@10"]
            dmax = float((h - pq[hs].reindex(h.index)).abs().max())
            print(f"  {ds}: alpha=0.5 vs {hs}: max|diff|={dmax:.2e}", flush=True)
        df.to_csv(fo, index=False)
        pickle.dump(top10, open(os.path.join(CACHE4, f"fused_top10_{model}_{ds}.pkl"), "wb"))
        print(f"{model} {ds}: {len(qrels)} q in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    models = sys.argv[1:] or [m for m in c2.DENSE_MODELS if any(c2.has_run(d, m) for d in c2.ALL_DATASETS)]
    for m in models:
        run(m)
