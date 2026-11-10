#!/usr/bin/env bash
# Agent 4 improvement pipeline (E1-E4). Idempotent; datasets / dense models are discovered from the files present.
# After the student's Colab zips are unpacked:
#   phase2_dense_runs.zip -> phase2/03_experiments/colab_runs/           (run_<m>_<ds>.npz, perquery_*.csv)
#   phase2_ce_runs.zip    -> phase2/03_experiments/colab_runs/ce/        (ce_<ds>.pkl; full-scale cross-encoder)
# run first Agent 3's pipeline (features for BGE-s/BGE-b, Tier B on 8 datasets), then this script:
#   cd /home/user/ir_lexical_gap && nohup bash phase2/03_experiments/run_all.sh > phase2/03_experiments/logs/run_all.log 2>&1 &&
#   nohup bash phase2/04_improvements/run_all4.sh > phase2/04_improvements/logs/run_all4.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export OMP_NUM_THREADS=1
PY=${PY:-python}
if [ -d ../03_experiments/colab_runs/colab_runs/ce ]; then mkdir -p ../03_experiments/colab_runs/ce && mv ../03_experiments/colab_runs/colab_runs/ce/* ../03_experiments/colab_runs/ce/; fi
echo "[1/5] alpha grid (every dense model with runs)"; $PY alpha_grid.py > logs/alpha_grid.log 2>&1
echo "[2/5] E1 per-query alpha";  for m in minilm bge bgebase; do $PY e1_alpha_router.py $m > logs/e1_$m.log 2>&1 || echo "  E1 $m skipped (inputs missing)"; done
echo "[3/5] E4 diagnostics";      $PY e4_why_routing_fails.py > logs/e4.log 2>&1
echo "[4/5] E2/E3 cross-encoder (CPU sample cache or Colab full-scale cache, whichever covers more queries)"
$PY ce_eval.py > logs/ce_eval.log 2>&1
echo "[5/5] F4 summary + figures"; $PY summarize4.py | tee logs/summarize4.log
echo "done: see results/summary4.md; update RESULTS.md narrative by hand if numbers changed."
