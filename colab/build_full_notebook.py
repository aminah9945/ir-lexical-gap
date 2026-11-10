"""Builds colab/lexical_gap_full_pipeline.ipynb: the complete project as one Colab notebook.
Every source file is embedded as a visible %%writefile cell, so the notebook is both readable and runnable."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = "/content/proj"


def src(rel):
    return open(os.path.join(ROOT, rel)).read()


def md(t):
    return ("markdown", t)


def code(t):
    return ("code", t)


def wf(rel, title):
    return [md(f"#### `{rel}` - {title}"), code(f"%%writefile {P}/{rel}\n" + src(rel))]


cells = [
    md("""# Diagnosing Lexical-Matching Limitations Across Domains: full pipeline

This notebook reproduces **every** experiment, statistical test, figure and the final PDF report.

**How to run**
1. *Runtime > Change runtime type > **T4 GPU*** (only the dense-retrieval step needs it).
2. *Runtime > Run all*. Expected time: **about 1-1.5 h** on a free T4 (lexical runs on Quora/Touche are CPU-bound).
3. The last cell downloads `report.pdf` and `results.zip`.

**Structure**
| Part | What it does | Proposal objective / hypothesis |
|---|---|---|
| 0 | Setup and data download (8 BEIR datasets) | - |
| 1 | Source code (each module in its own cell) | - |
| 2 | E1/E2/E5: Boolean, TF-IDF, BM25, preprocessing ablation, RM3, WordNet | Objective 2, H1 |
| 3 | E6: dense (MiniLM, bge-small) and hybrid retrieval | Objective 5, H5 |
| 4 | E3: vocabulary-overlap and jargon features | Objective 3 |
| 5 | Hypothesis tests H1, H2, H3, H5, H6 + figures | Objectives 2-5 |
| 6 | E7: failure-mode taxonomy H4 + manual audit | Objective 4 |
| 7 | Metric validation and PDF report | Deliverables |

Each verdict comes from a decision rule written in the code (docstrings of `analysis.py` and `failures.py`) before the results were inspected.
Seed = 42. BM25 k1 = 0.9, b = 0.4 (Pyserini BEIR defaults)."""),
    md("## 0. Setup"),
    code(f"""!nvidia-smi -L || echo "No GPU: the dense step will be very slow"
!pip -q install sentence-transformers PyStemmer wordfreq pytrec-eval-terrier reportlab statsmodels
import os, nltk
for d in ["src", "data", "report", "colab", "results", "figures", "cache"]:
    os.makedirs("{P}/" + d, exist_ok=True)
nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
os.chdir("{P}")
print("ok")"""),
    code(f"""%%bash
# BEIR datasets (CC BY-SA 4.0) from the official mirror
cd {P}/data
for d in nfcorpus scifact fiqa scidocs arguana trec-covid quora webis-touche2020; do
  if [ ! -f $d/corpus.jsonl ]; then wget -q https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$d.zip && unzip -q -o $d.zip && rm $d.zip; fi
  echo "$d: $(wc -l < $d/corpus.jsonl) docs"
done"""),
    md("## 1. Source code\nRun these cells to write the modules to disk. Reading them is the code review."),
]
cells += wf("src/common.py", "data loading, tokenisation/indexing, BM25/TF-IDF weighting, search, evaluation metrics")
cells += wf("src/run_lexical.py", "E1 classical models, E2 preprocessing ablation, E5 query expansion, BM25 sensitivity grid")
cells += wf("src/run_dense.py", "E6 dense retrieval and BM25+dense hybrids")
cells += wf("src/features.py", "E3 per-query overlap features and per-domain specificity statistics")
cells += wf("src/analysis.py", "H1, H2, H3, H5, H6 tests with fixed decision rules; figures 1-5")
cells += wf("src/failures.py", "H4 heuristic failure-mode taxonomy; figure 6")
cells += wf("src/validate_metrics.py", "cross-check of our metrics against pytrec_eval")
cells += wf("src/build_report.py", "assembles the PDF")
cells += wf("report/narrative.py", "report text; all numbers are read from results/")
cells += [
    md("#### Manual audit of failure labels (human judgements)\nThese files hold the analyst's manual judgements of sampled failure labels. "
       "They cannot be produced by code, so they are embedded as data. Development round (30 queries, rules v1, before the rules were revised) "
       "and test round (45 fresh queries, rules v2). Samples were drawn with `random_state=42` (dev) and `2026` (test) from `results/failure_labels.csv`."),
    code(f"%%writefile {P}/results/failure_manual_check_dev.csv\n" + src("results/failure_manual_check_dev.csv")),
    code(f"%%writefile {P}/results/failure_manual_check_test.csv\n" + src("results/failure_manual_check_test.csv")),
    code(f"%%writefile {P}/results/failure_audit_summary.json\n" + src("results/failure_audit_summary.json")),
    md("## 2. Lexical retrieval (E1, E2, E5)\nBoolean AND/OR, TF-IDF/VSM and BM25 on 4 preprocessing variants; RM3 and WordNet expansion; "
       "BM25 k1/b grid (tuned on test, reported only as an upper bound). The BM25 numbers should match Pyserini's BEIR references to within about 0.005."),
    code("!python src/run_lexical.py 2>&1 | grep -v -i warn"),
    md("## 3. Dense and hybrid retrieval (E6)\nall-MiniLM-L6-v2 and bge-small-en-v1.5, 256 tokens, exact cosine search; hybrids by RRF (k = 60) and by convex combination (alpha = 0.5)."),
    code("""%env DEVICE=cuda
%env BATCH=256
%env MAXLEN=256
!python src/run_dense.py minilm 2>&1 | grep -v -i "warn\\|loading weights"
!python src/run_dense.py bge 2>&1 | grep -v -i "warn\\|loading weights\""""),
    md("## 4. Overlap and specificity features (E3)\nPrimary overlap measure (fixed in advance): **coverage** = share of the query's distinct terms found in a relevant document, averaged over relevant documents. "
       "Specificity is *measured* as the share of tokens with general-English Zipf frequency < 3."),
    code("!python src/features.py 2>&1 | grep -v -i warn"),
    md("## 5. Hypothesis tests (H1, H2, H3, H5, H6)"),
    code("!python src/analysis.py 2>&1 | grep -v -i warn"),
    code("""import json, pandas as pd
from IPython.display import Image, display
V = json.load(open("results/verdicts.json"))
display(pd.DataFrame([(k, v["verdict"]) for k, v in V.items() if v], columns=["hypothesis", "verdict"]))
t = pd.read_csv("results/table_main.csv").pivot(index="system", columns="dataset", values="ndcg@10").round(3)
display(t)
display(pd.read_csv("results/h1_tests.csv").round(4))"""),
    code("""for f in ["fig1_heatmap", "fig2_domain", "fig3_overlap_bins", "fig4_forest", "fig5_ablation"]:
    display(Image(f"figures/{f}.png", width=900))"""),
    md("## 6. Failure-mode taxonomy (H4)\nBM25 failures (nDCG@10 < 0.1) are labelled AMBIG / ABBREV / MORPH / SYN / PARAPHRASE / OTHER by rules, "
       "then checked against the manual audit embedded above."),
    code("!python src/failures.py 2>&1 | grep -v -i \"warn\\|big\\|share =\""),
    code("""display(pd.read_csv("results/failure_crosstab.csv"))
display(pd.read_csv("results/failure_rescue_rates.csv").round(2))
print(json.load(open("results/h4_verdict.json")))
print(json.load(open("results/failure_audit_summary.json")))
display(Image("figures/fig6_failures.png", width=900))"""),
    md("## 7. Metric validation and PDF report"),
    code("""!python src/validate_metrics.py nfcorpus scifact fiqa | tee results/metric_validation.txt
!python src/build_report.py"""),
    code(f"""!cd {P} && zip -qr /content/results.zip results figures src report/narrative.py -x "*/__pycache__/*"
from google.colab import files
files.download("{P}/report/report.pdf")
files.download("/content/results.zip")"""),
]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s,
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells]}
out = os.path.join(ROOT, "colab", "lexical_gap_full_pipeline.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
