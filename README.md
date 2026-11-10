# Choose Per Collection, Not Per Query

Code, results and pre-registrations for the paper *"Choose Per Collection, Not Per Query: Selecting Between Sparse, Dense and Hybrid Retrieval on BEIR"* (Aminah Ali, FAST-NUCES Islamabad).

## In one paragraph
On eight BEIR collections, we compare BM25, three dense retrievers (MiniLM, bge-small, bge-base) and a 50/50 hybrid of the two.

- **The hybrid is a safe default.** With bge-base it beats BM25 on 7 of 8 collections (pooled +0.097 nDCG@10).
- **Per-query routing does not beat always using the hybrid.** We tried 18 versions of a router built on query-performance-prediction signals, plus a pre-registered router that reads the query text.
- **Why.** The signals measure how hard a query is for every system, not which system suits it.
- **What works.** Choosing once per collection: setting the hybrid weight from 20 labelled queries gains +0.021 (exploratory).

## Layout
| Folder | Contents |
|---|---|
| `src/` | Phase 1: data loading, BM25 / TF-IDF / Boolean, metrics, features, hypothesis tests H1-H6 |
| `results/`, `figures/`, `report/` | Phase 1 results and report |
| `colab/`, `phase2/colab/` | Google Colab notebooks for the GPU dense encodings |
| `phase2/03_experiments/` | QPP router, effect sizes, BM25 LODO tuning, pooling bias (`run_all.sh`, `EXPERIMENT_LOG.md` with the pre-declarations) |
| `phase2/04_improvements/` | Pre-declared improvement tests P1-P4 (`run_all4.sh`) |
| `phase2/05_alpha_fewshot/` | Few-label hybrid-weight tuning (exploratory) |
| `phase2/06_text_router/` | Pre-registered text-based router (E5) |
| `phase2/07_disagreement/` | Pre-registered disagreement-guided labelling (E7) |
| `phase2/07_paper/` | Paper source (`main.tex`); `numbers_trace.md` maps every number in the paper to its source file |
| `preregistrations/` | Pre-registration texts and their timestamps |
| `laptop_kit/` | Script to encode with a larger model (bge-large) on a local GPU |

## Reproduce
```bash
pip install -r requirements.txt
python scripts/download_beir.py          # 8 BEIR test collections into data/
python src/run_lexical.py                # BM25 / TF-IDF / Boolean runs into cache/
# Dense encodings need a GPU: run phase2/colab/phase2_dense_runs.ipynb in Colab,
# then unzip its output into phase2/03_experiments/colab_runs/
OMP_NUM_THREADS=1 bash phase2/03_experiments/run_all.sh
OMP_NUM_THREADS=1 bash phase2/04_improvements/run_all4.sh
python phase2/05_alpha_fewshot/run.py
python phase2/06_text_router/train.py && python phase2/06_text_router/eval.py
python phase2/07_disagreement/run.py
```

Notes:
- The full pipeline takes about 60-90 minutes on 4 CPU cores. The text router takes about 4 hours on CPU.
- Seed 42 throughout. Significance uses 10^4 sign flips; confidence intervals use 10^4 bootstrap resamples.
- Per-query result files are included, so the analyses can be checked without rerunning retrieval.

## Use of AI tools
This work was carried out with extensive use of an AI assistant (Anthropic Claude), under the author's direction, as stated in the paper's acknowledgment. The assistant designed and ran the experiments, wrote the code and drafted the paper. The author directed the work and takes responsibility for it.
