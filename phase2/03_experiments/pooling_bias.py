"""W9: pooling-bias check. judged@k = share of the top-k documents that have ANY qrel entry (any grade, incl. 0)
for the query; and condensed-list nDCG@10 (unjudged documents removed before evaluation; a standard robustness check).

Systems: B = bm25, every dense model with a ranking on disk (Phase-1 pickles or Colab npz), H(D) = fixed convex
hybrid (Phase-1 pickle if present, else computed with common2.convex), and bm25lodo if present.
Outputs results/w9_judged.csv (per dataset x system) and results/w9_judged_perquery.csv.gz
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import common2 as c2
from common2 import common


def read_qrels(ds):
    q = defaultdict(dict)
    with open(os.path.join(common.DATA, ds, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            a, d, s = line.rstrip("\n").split("\t")
            q[a][d] = int(s)
    return {k: v for k, v in q.items() if any(x > 0 for x in v.values())}


def systems_for(ds):
    out = {"bm25": lambda: c2.load_run(ds, "bm25")}
    for m in c2.DENSE_MODELS:
        if c2.has_run(ds, m):
            out[m] = (lambda m=m: c2.load_run(ds, m))
            if c2.has_run(ds, f"hybrid_cc_{m}"):
                out[f"hybrid_cc_{m}"] = (lambda m=m: c2.load_run(ds, f"hybrid_cc_{m}"))
            else:
                out[f"hybrid_cc_{m}"] = (lambda m=m: c2.convex(c2.load_run(ds, "bm25"), c2.load_run(ds, m)))
    if c2.has_run(ds, "bm25lodo"):
        out["bm25lodo"] = lambda: c2.load_run(ds, "bm25lodo")
    return out


def main():
    rows = []
    for ds in c2.ALL_DATASETS:
        qrels = read_qrels(ds)
        for sysname, loader in systems_for(ds).items():
            run = loader()
            for q, rels in qrels.items():
                ranked = [d for d, _ in run.get(q, [])]
                j10 = np.mean([d in rels for d in ranked[:10]]) if ranked else 0.0
                j100 = np.mean([d in rels for d in ranked[:100]]) if ranked else 0.0
                full = common.evaluate_query(ranked, rels)["ndcg@10"]
                cond = common.evaluate_query([d for d in ranked if d in rels], rels)["ndcg@10"]
                rows.append((ds, sysname, q, j10, j100, full, cond))
            del run
        print(ds, "done", flush=True)
    pq = pd.DataFrame(rows, columns=["dataset", "system", "qid", "judged10", "judged100", "ndcg10", "ndcg10_condensed"])
    pq.to_csv(os.path.join(c2.RES2, "w9_judged_perquery.csv.gz"), index=False)
    agg = pq.groupby(["dataset", "system"])[["judged10", "judged100", "ndcg10", "ndcg10_condensed"]].mean().reset_index()
    agg["condensed_gain"] = agg.ndcg10_condensed - agg.ndcg10
    agg["n"] = pq.groupby(["dataset", "system"]).size().values
    agg.to_csv(os.path.join(c2.RES2, "w9_judged.csv"), index=False)
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
