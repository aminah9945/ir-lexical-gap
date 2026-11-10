# Run order: reproduce everything in Colab from Google Drive

All notebooks are in `MyDrive/IR_Lexical_Gap/01_notebooks/`. To open one, right-click it in Drive and choose *Open with > Google Colaboratory*. Each notebook mounts Drive in its first code cell; allow access when asked. Everything you produce is written to `MyDrive/IR_Lexical_Gap/`, so nothing is lost when Colab disconnects.

| # | Notebook | Runtime | Time (approx.) | Writes to Drive |
|---|---|---|---|---|
| 0 | `00_SETUP_drive.ipynb` | CPU | 10-20 min | `data/<8 datasets>/`; `04_literature/pdfs/<topic>/*.pdf` + `download_report.txt`; empty `project/` layout |
| 1 | `phase1_full_pipeline_drive.ipynb` (runs Phase-1 `lexical_gap_full_pipeline.ipynb`) | T4 GPU | 1-1.5 h | `project/results/`, `project/cache/` (rankings, about 2 GB), `project/figures/`, `project/report/report.pdf` |
| 2 | `phase2_dense_runs.ipynb` | T4 GPU | 1.5-2 h | `colab_runs/run_<model>_<ds>.npz`, `perquery_*.csv` |
| 3 | `phase2_ce_rerank.ipynb` | T4 GPU | 45-75 min | `colab_runs/ce/ce_<ds>.pkl` |
| 4 | `phase2_analysis.ipynb` (Agents 3 + 4) | CPU | 2-4 h | `project/phase2/03_experiments/{results,figures,logs,cache}`, `project/phase2/04_improvements/{...}` |

## Steps

### Step 0: setup (once)
1. Open `00_SETUP_drive.ipynb`.
2. Click *Runtime > Run all*.
3. At the end, check that it prints all 8 datasets and the PDF count per topic.
4. Papers marked "DOI only" in `04_literature/reading_list.md`, plus any download that failed (see `04_literature/pdfs/download_report.txt`), are paywalled. Download them from the university network and drop them into the topic folder shown in the report.

### Step 1: Phase-1 full pipeline
1. Open `phase1_full_pipeline_drive.ipynb`. You do not need to open `lexical_gap_full_pipeline.ipynb`; the runner executes it for you.
2. Set *Runtime > Change runtime type > T4 GPU*.
3. Click *Runtime > Run all*.
   - The runner links `/content/proj/{results,cache,figures,report}` into `MyDrive/IR_Lexical_Gap/project/` and `/content/proj/data` to `MyDrive/IR_Lexical_Gap/data`.
   - It then executes every code cell of `lexical_gap_full_pipeline.ipynb`, unchanged, read from `01_notebooks`.
   - This is what makes the Phase-1 outputs available to step 4.
   - The dataset-download cell is skipped because the data is already present.
   - If you run `lexical_gap_full_pipeline.ipynb` directly, it still works, but its outputs stay in the temporary runtime and are lost.
   - Lexical runs on Quora/Touche are CPU-bound and take a while.
4. At the end it downloads `report.pdf` and `results.zip` to your computer. Copies also stay in Drive under `project/`.
5. If a cell fails, the runner prints its number. Set `START` to that number in the runner's loop cell and run that cell again.
6. If Colab disconnects, run all again. Phase 1 recomputes from scratch, except the dense embeddings that are already cached in Drive.
7. A copy of the regenerated report is saved as `05_reports_and_paper/report_regenerated.pdf`.

### Step 2: dense runs for Phase 2
1. Open `phase2_dense_runs.ipynb`.
2. Set the runtime to T4 GPU.
3. Click *Run all*.
4. The notebook is resumable: if it disconnects, run all again and finished (model, dataset) pairs are skipped.
   - On 2026-10-22, `colab_runs/` already held your earlier runs: all 8 datasets for MiniLM and BGE-s, and 7 of 8 for BGE-b (webis-touche2020 was still missing).
   - Re-running therefore only computes what is missing.
5. Its last cell zips `colab_runs` for download. The zip is only needed if you also want to send the runs to the Claude chat; the analysis notebook reads them straight from Drive.

### Step 3: cross-encoder re-ranking
1. Open `phase2_ce_rerank.ipynb`.
2. Set the runtime to T4 GPU.
3. Click *Run all*. It is resumable per dataset.
4. Run it after step 2, so that the hybrid candidate pools exist. Without step 2, it scores only the BM25 top-100.

### Step 4: Phase-2 analysis
1. Open `phase2_analysis.ipynb`.
2. Use a CPU runtime; no GPU is needed.
3. Click *Run all*.
   - Section 2 (prerequisite check) lists what is present per dataset.
   - `data`, `perquery_lexical`, `features` and `bm25 run` must be `True` for all 8 datasets, which means step 1 ran through the Drive runner.
   - Without steps 2 and 3, the models and cross-encoder parts whose inputs are missing are skipped.
4. It runs Agent 3's `phase2/03_experiments/run_all.sh`, then Agent 4's `phase2/04_improvements/run_all4.sh`. The code is copied from `02_code/` on Drive, and its md5 sums are checked against the built version.
5. It writes all CSVs, `summary_tables.md`, `summary4.md` and **all Phase-2 figures (PNG) into Drive**. It then shows them in the notebook and flushes Drive.

## Honest limits
- **RAM.** Phase-2 analysis was developed on a 15 GB RAM machine. Free Colab has about 12.7 GB, and the Quora/Touche top-1000 rankings are large. Peak RAM in Colab was not measured.
  - If Colab reports "session crashed after using all available RAM", run all again. The W6 BM25 grid, the QPP features and the alpha grid are cached.
  - If it crashes again at the same step, use a High-RAM runtime or run the two `.sh` scripts on a 16 GB PC.
- **Numbers.** When you regenerate the rankings yourself, the MiniLM/hybrid Phase-1 runs exist for all 8 datasets. Locally they existed for 5. Some Phase-2 tables can therefore differ slightly from `RESULTS_SUMMARY.md`. The narrative `.md` files are not regenerated automatically.
- **Optional notebook.** `lexical_gap_dense_gpu.ipynb` is the older dense-only notebook. The full pipeline (step 1) supersedes it, so you do not need to run it.

## Drive folder map
```
MyDrive/IR_Lexical_Gap/
  README.txt, 01_RUN_ORDER.md, manifest.md
  01_notebooks/         00_SETUP_drive, phase1_full_pipeline_drive, lexical_gap_full_pipeline, phase2_dense_runs,
                        phase2_ce_rerank, phase2_analysis (+ the older lexical_gap_dense_gpu)
  02_code/              src/ (Phase 1), colab/ (Phase-1 builders), phase2/ (Agent 3 + Colab scripts), phase2/improvements/ (Agent 4)
  03_results/           phase1/ and phase2/ result tables (small CSV/MD copies)
  04_literature/        references.bib, reading_list.md, pdfs/<topic>/ (filled by step 0)
  05_reports_and_paper/ report PDF, phase2_agent_docs/ (00-04 md, RESULTS_SUMMARY, EXPERIMENT_LOG, ...)
  data/                 8 BEIR datasets (filled by step 0)
  colab_runs/           dense + cross-encoder runs (filled by steps 2-3)
  project/              mirror of the local project, filled by steps 1 and 4: src/, results/, cache/, figures/,
                        report/, phase2/03_experiments/..., phase2/04_improvements/...  (data/ lives one level up)
```
