# PENDING.md: paper parts that depend on the student's pending Colab runs

Each item below matches a `% PENDING-COLAB:` comment in `main.tex`. To find them all, run `grep -n PENDING-COLAB main.tex`.

## Inputs still to come

1. **`phase2_dense_runs.zip`** from `phase2/colab/phase2_dense_runs.ipynb`. It contains MiniLM, BGE-s and BGE-b (bge-base-en-v1.5) rankings on all 8 datasets. Unzip into `phase2/03_experiments/colab_runs/`.
2. **`phase2_ce_runs.zip`** from `phase2/colab/phase2_ce_rerank.ipynb`. It contains full-scale cross-encoder scores. Unzip into `phase2/03_experiments/colab_runs/ce/`.

Then rerun both pipelines (command from `04_interpretation.md` §5):

```
cd /home/user/ir_lexical_gap && nohup bash phase2/03_experiments/run_all.sh > phase2/03_experiments/logs/run_all.log 2>&1 && \
  nohup bash phase2/04_improvements/run_all4.sh > phase2/04_improvements/logs/run_all4.log 2>&1 &
```

After rerunning, regenerate the figures (`python phase2/07_paper/make_fig_indomain.py`, `python phase2/07_paper/make_fig_fewshot.py`) and the revision numbers (`python phase2/07_paper/analysis_extra/extra_numbers.py`). Table IV CIs and Table V CIs come from that script.

(Revised 2026-10-22 after the 08 critique. The win-share and router-bar figures were removed, so their items no longer apply.)

## Items in `main.tex`

| # | Location in main.tex | What changes | New source file(s) |
|---|---|---|---|
| 1 | Abstract (comment before `\end{abstract}`) | Re-check the router 0.525 vs 0.534, the six-collection +0.004 [−0.001, +0.008], the few-shot +0.010 to +0.014, and the "On five collections" scope against the 8-dataset / BGE-b / full-CE results. | `A3/w1_router_macro.csv`, `A4/f4_primary.csv`, `A4/e4_fewshot_dataset_selection.csv` |
| 2 | Intro, contribution 1 ("seven configurations") | The number of router configurations grows: Tier A BGE-b, and Tier B for MiniLM/BGE-s/BGE-b on 8 datasets. | `A3/w1_router_macro.csv` |
| 3 | Sec. III-B Dense | Add bge-base-en-v1.5 ("BGE-b", 110M), cite `hf_bgebase`. Add that key back to `references.bib` from `phase2/literature/references.bib`; it was trimmed because it is not cited yet. | — |
| 4 | Sec. III-C Tier B sentence ("currently available for MiniLM on the five datasets") | Tier B becomes available for all three dense models on 8 datasets. | `A3/qpp/tierB_*` |
| 5 | Table `tab:main` | Add BGE-b and H(BGE-b) columns; possibly shrink to `table*`. | `A3/perquery/` or `P1/`-style per-query CSV from Colab |
| 6 | Table `tab:effects` | Add BGE-b − BM25, H(BGE-b) − BM25 and H(BGE-b) − BGE-b rows. | `A3/w7_effect_sizes.csv` |
| 7 | Table `tab:router` | Add the Tier A BGE-b (8) row. Add Tier B MiniLM/BGE-s/BGE-b (8) rows; these replace the Tier B MiniLM (5) row. Update the Holm ± and mix values. | `A3/w1_router_macro.csv`, `A3/w1_router_tests.csv` |
| 8 | Table `tab:improve` and the P1/P2 text | P1 and P2 extend to 8 datasets and D ∈ {BGE-s, BGE-b}. The F4 family stays as declared: P1-P3 keep MiniLM as the primary model; the others are secondary. | `A4/f4_primary.csv`, `A4/e1_*`, `A4/e2_*` |
| 9 | P3 paragraph (CE) | Full-scale CE on all queries (Quora about 2.1k) replaces the ≤150-query CPU sample. Update −0.040, −0.120 (ArguAna), +0.025, +0.143, −0.164 [−0.230, −0.100] and −0.059. Remove the "at most 150 queries" sentence. | `A4/e3_ce_perdataset.csv`, `A4/e3_ce_tests.csv` |
| 10 | P4 paragraph | Add the in-domain ceiling for BGE-b (Tier A) and for the Tier B BGE configurations. | `A4/e4_indomain_tests.csv` |
| 11 | Sec. IV-E few-shot paragraph and Fig. `fig:fewshot` | Add the BGE-b curve. `make_fig_fewshot.py` needs a third model line, mirroring `summarize4.py`. | `A4/e4_fewshot_dataset_selection.csv`, `A4/e4_splithalf.csv` |
| 12 | Limitations: "Model strength" bullet | Revise once BGE-b and the full-scale CE are in. Until then, do **not** claim that a stronger dense model leaves the routing conclusion unchanged. | — |
| 14 | Table `tab:sens` (small-collection sensitivity) | Rows 1–2 and 6–7 change with BGE-b and Tier B on 8 collections; row 4 changes with the full-scale CE. Recompute with `analysis_extra/extra_numbers.py` (add the new configs to its lists). | `analysis_extra/extra_numbers.json` |
| 13 | Limitations: "Judgement pools" bullet and the Discussion "Prior shift" bullet | Add dense-side judged@10/100 on TREC-COVID, Touché and Quora. That is the decisive pool-bias evidence. | `A3/w9_judged.csv` |

## Non-Colab items noted while writing (for the critic or the student)

- N1. **Headroom range.** The paper's "23–87% of headroom" and "2–27% of queries" are now stated for BGE-s only. For MiniLM, the share of headroom spans 21–89% (`A4/e4_headroom_concentration.csv`).
- N2. **Boolean AND.** The brief says "empty sets for 62–93% of queries on 7/8". The file `P1/boolean_set_summary.csv` gives 6/8 in that range: NFCorpus is 40% and Touché 6%. The paper says "six of the eight".
- N3. **ArguAna tuned BM25.** The brief's "b = 0.9 reaches 0.529" comes from the Phase-1 grid, which used a 1,000-query subsample (`LOG3` §1). The paper uses the all-query W6 value: 0.528 at (k1, b) = (2.5, 0.9).
- N4. **References not in the bib.**
  - Holm (1979) is excluded from the verified bib, so Holm's procedure is named without a citation.
  - The cross-encoder checkpoint and `wordfreq` are named without a citation; no verified entry exists.
  - Add these only after verifying them.
- N5. **Author block and acknowledgement** are placeholders. The AI-use disclosure is now written (revision); the student should confirm it.
- N6. **Repository URL** in the reproducibility statement is a placeholder.

---

## Resolution (2026-11-03)

All 16 `% PENDING-COLAB` markers were resolved from the files produced by the 2026-10-22 full Colab ingest (3 dense models x 8 datasets, Tier B for all three). Numbers: `numbers_trace.md` §"2026-11-03 update"; CIs: `analysis_extra/extra_numbers_v2.json`, `analysis_extra/p2_rest_v2.json`.

| # | Status |
|---|---|
| 1 Abstract | updated (BGE-b, 18 configs, P1 on 8, few-shot BGE-b) |
| 2 Contribution count | "seven" -> 18 configurations |
| 3 BGE-b in systems + bib | done; `hf_bgebase` added to `references.bib` |
| 4 Tier B on 8 | done |
| 5 tab:main | BGE-b and H(BGE-b) columns added (single column, \resizebox) |
| 6 tab:effects | 3 BGE-b rows added |
| 7 tab:router | rows Tier A BGE-b, Tier B x3 (8), Tier A-dagger BGE-b (8); Tier B MiniLM (5) and Tier A-dagger MiniLM (5) rows removed (superseded) |
| 8 tab:improve, P1/P2 | P1 now on 8 collections (MiniLM re-encoded): -0.0014 (was -0.0069 in the interim paper). P2 kept at its declared 5-collection scope (-0.0044, reproduced exactly); 6-collection and BGE secondaries added |
| 9 P3 full-scale CE | NOT RUN. Paper now states the CE sample is the only CE evidence. |
| 10 P4 BGE-b | added |
| 11 few-shot BGE-b | added to text and figure (`make_fig_fewshot.py` updated; old figure kept as `figs/fig_fewshot_submitted.png`) |
| 12 Model-strength bullet | revised |
| 13 judged@k dense | added (from `A3/w9_judged.csv`) |
| 14 tab:sens | BGE-b rows added; CE row unchanged (no full-scale CE) |

Bugs found in `04_improvements/summarize4.py` (not fixed in code, worked around):
- P1 row hard-codes k=5, but `e1_perquery_minilm` now has 8 collections.
- P2 MACRO over "rest" is NaN with a spurious p=0.0001 because TREC-COVID and Touche have no held-out queries; this also feeds a wrong p into the F4 Holm step. Correct P2 recomputed in `analysis_extra/p2_rest_v2.py` (matches per-dataset deltas exactly).
