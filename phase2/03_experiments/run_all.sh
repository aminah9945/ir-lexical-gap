#!/usr/bin/env bash
# Phase-2 experiment pipeline (Agent 3). Re-run after unpacking the student's Colab zip into
#   phase2/03_experiments/colab_runs/   (files run_<model>_<ds>.npz, perquery_<model>_<ds>.csv)
# Everything is resumable / idempotent: datasets and models are discovered from the files present.
# Usage: bash phase2/03_experiments/run_all.sh            (from the project root or anywhere)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export OMP_NUM_THREADS=1
PY=${PY:-python}

# Colab zip layout is colab_runs/<files> (zip of the colab_runs folder); flatten a nested colab_runs/colab_runs/.
if [ -d colab_runs/colab_runs ]; then mv colab_runs/colab_runs/* colab_runs/ && rmdir colab_runs/colab_runs; fi

echo "[1/8] Colab ingest + sanity gates";         $PY colab_ingest.py            | tee logs/colab_ingest.log
echo "[2/8] W6 BM25 grid / LODO selection / runs"; $PY bm25_lodo.py all           > logs/w6_all.log 2>&1
echo "[3/8] W1/W2 query-time features";           $PY qpp_features.py            > logs/qpp_features_all.log 2>&1
echo "[4/8] W7 effect sizes";                      $PY effect_sizes.py ndcg@10    > logs/w7.log 2>&1
                                                   $PY effect_sizes.py recall@100 >> logs/w7.log 2>&1
echo "[5/8] W3 gap model (+W2 proxies)";           $PY gap_model.py               > logs/w3.log 2>&1
echo "[6/8] W1 QPP evaluation";                    $PY qpp_eval.py                > logs/qpp_eval.log 2>&1
echo "[7/8] W1 routers (parallel, 3 at a time)"
$PY router.py list | tr ' ' '\n' | xargs -P 3 -I{} sh -c "$PY router.py {} > logs/router_{}.log 2>&1 || true"
$PY router.py collect
echo "[8/8] W9 pooling bias + summary tables";     $PY pooling_bias.py            > logs/w9.log 2>&1
$PY fig_winshare.py > logs/fig.log 2>&1
$PY summarize.py | tee logs/summary.log
echo "done. See results/ and RESULTS_SUMMARY.md (regenerate the narrative parts by hand if numbers changed)."
