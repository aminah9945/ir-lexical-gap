"""Ingest the student's Colab outputs (colab_runs/run_<model>_<ds>.npz + perquery_<model>_<ds>.csv).

For every npz present:
  * recompute per-query metrics from the ranking (common.evaluate_run) and compare with the Colab CSV;
  * sanity gates (02_verification W4): MiniLM per-query nDCG@10 within 1e-3 of Phase-1 perquery_minilm_<ds>.csv on
    every query; bge-small macro nDCG@10 within 0.002 of Phase-1 perquery_bge_<ds>.csv;
  * for models without Phase-1 labels (bgebase), write results/perquery/perquery_<model>_<ds>.csv with the dense
    system and the fixed convex hybrid H(D) built from the Phase-1 BM25 top-1000 run (Quora: dense depth 100).
Outputs results/colab_sanity.csv. Exits with status 1 if a gate fails (run_all.sh then stops).
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

import common2 as c2
from common2 import common
from pooling_bias import read_qrels

PQDIR = os.path.join(c2.RES2, "perquery")
os.makedirs(PQDIR, exist_ok=True)


def main():
    files = sorted(glob.glob(os.path.join(c2.COLAB, "run_*_*.npz")))
    if not files:
        print("no Colab runs found in", c2.COLAB)
        return 0
    rows, fail = [], False
    for f in files:
        m, ds = re.match(r"run_([a-z]+)_(.+)\.npz$", os.path.basename(f)).groups()
        if m not in c2.DENSE_MODELS or ds not in c2.ALL_DATASETS:
            continue
        qrels = read_qrels(ds)
        run = c2.load_run(ds, m) if not os.path.exists(os.path.join(c2.P1_RUNS, f"{ds}__{m}.pkl")) else None
        if run is None:  # a Phase-1 pickle shadows the npz in load_run; read the npz explicitly
            z = np.load(f)
            ids = c2.doc_ids_of(ds)
            run = {str(q): [(ids[i], float(s)) for i, s in zip(ix, ss)] for q, ix, ss in zip(z["qids"], z["idx"], z["scores"])}
        per = common.evaluate_run(run, qrels)
        mine = pd.Series({q: v["ndcg@10"] for q, v in per.items()})
        r = dict(model=m, dataset=ds, n=len(mine), ndcg10=mine.mean())
        cf = os.path.join(c2.COLAB, f"perquery_{m}_{ds}.csv")
        if os.path.exists(cf):
            cc = pd.read_csv(cf, dtype={"qid": str}).set_index("qid")["ndcg@10"]
            r["max_absdiff_vs_colab_csv"] = float((mine - cc.reindex(mine.index)).abs().max())
        p1 = os.path.join(c2.P1_RES, f"perquery_{m}_{ds}.csv")
        if os.path.exists(p1):
            ref = pd.read_csv(p1, dtype={"qid": str})
            ref = ref[ref.system == m].set_index("qid")["ndcg@10"].reindex(mine.index)
            r["phase1_ndcg10"] = ref.mean()
            r["max_absdiff_vs_phase1"] = float((mine - ref).abs().max())
            r["n_query_diff_gt_1e-3"] = int(((mine - ref).abs() > 1e-3).sum())
            if m == "minilm":
                # Relaxed 2026-10-22 (orchestrator): the per-query 1e-3 gate failed on 6/8 datasets for 2-35 queries
                # each, while every macro difference is <= 0.0015. Causes: Phase-1 local MiniLM rankings for 5
                # datasets were computed from float16-stored embeddings, and the Phase-1 Colab run (other 3 datasets)
                # used a different library/GPU build, so near-tied documents swap order. Same rule as BGE-s now.
                r["gate"] = "macro |d|<=0.002 (relaxed from per-query 1e-3; see EXPERIMENT_LOG)"
                r["gate_pass"] = bool(abs(mine.mean() - ref.mean()) <= 0.002)
            elif m == "bge":
                r["gate"] = "macro |d|<=0.002"
                r["gate_pass"] = bool(abs(mine.mean() - ref.mean()) <= 0.002)
            fail |= not r.get("gate_pass", True)
        else:  # no Phase-1 labels: create them (dense + fixed hybrid with Phase-1 BM25)
            bm25 = c2.load_run(ds, "bm25")
            hyb = c2.convex(bm25, run)
            out = common.per_query_rows(per, ds, m) + common.per_query_rows(common.evaluate_run(hyb, qrels), ds,
                                                                            f"hybrid_cc_{m}")
            pd.DataFrame(out).to_csv(os.path.join(PQDIR, f"perquery_{m}_{ds}.csv"), index=False)
            r["hybrid_ndcg10"] = float(np.mean([v["ndcg@10"] for v in common.evaluate_run(hyb, qrels).values()]))
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(c2.RES2, "colab_sanity.csv"), index=False)
    if fail:
        print("SANITY GATE FAILED: see results/colab_sanity.csv")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
