#!/bin/bash
# Sequential dense jobs after the first batch (reproducibility record of what was run).
cd "$(dirname "$0")/.."
while pgrep -f "run_dense.py minilm nfcorpus" >/dev/null || pgrep -f run_lexical.py >/dev/null; do sleep 30; done
MAXLEN=256 python src/run_dense.py minilm quora            > results/log_dense_minilm_quora.txt 2>&1
MAXLEN=128 python src/run_dense.py minilm trec-covid webis-touche2020 > results/log_dense_minilm_large.txt 2>&1
MAXLEN=256 python src/run_dense.py bge nfcorpus scifact arguana scidocs fiqa > results/log_dense_bge.txt 2>&1
echo ALLDONE >> results/log_dense_bge.txt
