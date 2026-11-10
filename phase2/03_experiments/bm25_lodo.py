"""W6: BM25 (k1, b) selected without test leakage (leave-one-dataset-out).

Stages (resumable; each dataset processed alone to bound RAM):
  grid    full-query per-query nDCG@10 / recall@100 for k1 x b on every dataset
          -> results/w6_grid/grid_perquery_<ds>.csv.gz, results/w6_grid_macro.csv
  select  for each held-out h: (k1, b) maximising the macro nDCG@10 over the OTHER datasets
          -> results/w6_lodo_selection.csv
  run     BM25 with the selected (k1, b) on h, top-1000, saved as cache/runs/<h>__bm25lodo.pkl
          + per-query metrics results/perquery/perquery_bm25lodo_<h>.csv; and, where a dense run exists,
          the fixed convex hybrid with it (hybrid_cc_<m>_bm25lodo).
Usage: python bm25_lodo.py [grid|select|run|all] [datasets...]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import common2 as c2
from common2 import common

K1S = [0.6, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0]
BS = [0.2, 0.4, 0.6, 0.75, 0.9]
GDIR = os.path.join(c2.RES2, "w6_grid")
PQDIR = os.path.join(c2.RES2, "perquery")
os.makedirs(GDIR, exist_ok=True)
os.makedirs(PQDIR, exist_ok=True)


def grid(ds):
    out = os.path.join(GDIR, f"grid_perquery_{ds}.csv.gz")
    if os.path.exists(out):
        print("grid exists", ds)
        return
    t0 = time.time()
    idx = common.Index(ds)
    qids = list(idx.qrels)
    C, vocab, an = idx.variant(True, True)
    Q = common.query_matrix([idx.queries[q] for q in qids], vocab, an)
    rows = []
    for k1 in K1S:
        for b in BS:
            t = time.time()
            W, _ = common.bm25_matrix(C, k1, b)
            run = common.to_run(common.search(W, Q, k=100, batch=256), qids, idx.doc_ids)
            per = common.evaluate_run(run, idx.qrels)
            for q in qids:
                rows.append((q, k1, b, per[q]["ndcg@10"], per[q]["recall@100"]))
            del W, run
            print(f"  {ds} k1={k1} b={b} nDCG@10={np.mean([per[q]['ndcg@10'] for q in qids]):.4f} ({time.time()-t:.0f}s)",
                  flush=True)
    pd.DataFrame(rows, columns=["qid", "k1", "b", "ndcg10", "recall100"]).to_csv(out, index=False)
    print(f"grid {ds} done in {time.time()-t0:.0f}s", flush=True)


def macro_table():
    rows = []
    for ds in c2.ALL_DATASETS:
        p = os.path.join(GDIR, f"grid_perquery_{ds}.csv.gz")
        if not os.path.exists(p):
            continue
        g = pd.read_csv(p, dtype={"qid": str})
        m = g.groupby(["k1", "b"])[["ndcg10", "recall100"]].mean().reset_index()
        m["dataset"] = ds
        rows.append(m)
    t = pd.concat(rows)
    t.to_csv(os.path.join(c2.RES2, "w6_grid_macro.csv"), index=False)
    return t


def select():
    t = macro_table()
    dss = sorted(t.dataset.unique(), key=c2.ALL_DATASETS.index)
    piv = t.pivot_table(index=["k1", "b"], columns="dataset", values="ndcg10")
    rows = []
    for h in dss:
        others = [d for d in dss if d != h]
        score = piv[others].mean(1)
        k1, b = score.idxmax()
        best_test = piv[h].idxmax()
        rows.append(dict(dataset=h, k1=k1, b=b, train_macro=score.max(),
                         test_ndcg_selected=piv.loc[(k1, b), h], test_ndcg_default=piv.loc[(0.9, 0.4), h],
                         oracle_k1=best_test[0], oracle_b=best_test[1], test_ndcg_oracle_upper_bound=piv[h].max(),
                         n_train=len(others), grid_edge=(k1 == max(K1S))))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(c2.RES2, "w6_lodo_selection.csv"), index=False)
    print(s.round(4).to_string())
    return s


def run(ds):
    sel = pd.read_csv(os.path.join(c2.RES2, "w6_lodo_selection.csv")).set_index("dataset")
    k1, b = float(sel.loc[ds, "k1"]), float(sel.loc[ds, "b"])
    t = time.time()
    idx = common.Index(ds)
    qids = list(idx.qrels)
    C, vocab, an = idx.variant(True, True)
    Q = common.query_matrix([idx.queries[q] for q in qids], vocab, an)
    W, _ = common.bm25_matrix(C, k1, b)
    r = common.to_run(common.search(W, Q, k=1000, batch=256), qids, idx.doc_ids)
    qrels = idx.qrels
    del W, C, idx
    c2.save_run2(r, ds, "bm25lodo")
    rows = common.per_query_rows(common.evaluate_run(r, qrels), ds, "bm25lodo")
    for m in c2.DENSE_MODELS:
        if c2.has_run(ds, m):
            h = c2.convex(r, c2.load_run(ds, m))
            rows += common.per_query_rows(common.evaluate_run(h, qrels), ds, f"hybrid_cc_{m}_bm25lodo")
    pd.DataFrame(rows).to_csv(os.path.join(PQDIR, f"perquery_bm25lodo_{ds}.csv"), index=False)
    print(f"run {ds} k1={k1} b={b} done ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    dss = sys.argv[2:] or c2.ALL_DATASETS
    if stage in ("grid", "all"):
        for d in dss:
            grid(d)
    if stage in ("select", "all"):
        select()
    if stage in ("run", "all"):
        for d in dss:
            run(d)
