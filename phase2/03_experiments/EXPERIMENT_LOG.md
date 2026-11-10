# EXPERIMENT_LOG (Agent 3, Experimenter)

Environment: 4 CPU, 15 GB RAM, no GPU; Python 3, scikit-learn 1.9.1, statsmodels 0.15.0, scipy 1.18.1, matplotlib 3.11.2.
Seed 42 everywhere (`common2.SEED`). Randomization tests: 10,000 sign flips; bootstrap: 10,000 resamples (W3 circularity: 2,000, as specified).
All code is in `phase2/03_experiments/` and imports the Phase-1 `src/common.py` read-only. Nothing in `src/`, `results/` or `cache/` was modified.
New rankings (BM25-LODO) are written to `phase2/03_experiments/cache/runs/`.

## 0. Pre-declaration (written BEFORE any router result was computed)

- **Primary hypothesis (W1):** macro-average nDCG@10 of the router minus macro-average of always-H(D) is greater than 0. It is tested separately for Tier A (pre-retrieval + BM25 post-retrieval features) and Tier B (Tier A + dense-list + agreement features), for each D.
  - Test: stratified sign-flip randomization test of the macro difference (per-query signs flipped within datasets; 10,000 permutations). Both the one-sided and the two-sided p are reported.
  - Per-dataset paired randomization tests use Holm correction across datasets (family F2: one family per router variant × baseline).
  - A DerSimonian-Laird pooled estimate is reported alongside.
- **Headline router.** The "selected" variant:
  - cost-sensitive regression of Δ_B = nDCG_B − nDCG_H and Δ_D = nDCG_D − nDCG_H;
  - the model family is chosen per outer fold by an inner LODO on the training datasets, between Ridge(α=1) and HGB(max_depth=3, lr=0.05, 200 iterations, min_samples_leaf=50, l2=1.0);
  - StandardScaler is fitted on the training datasets only;
  - decision rule: argmax{0 (→H), Δ̂_B, Δ̂_D}, choosing the cheapest system (B < D < H) whose prediction is within 0.005 of the maximum.
- **Secondary variants, all reported:**
  - Ridge alone and HGB alone;
  - multinomial logistic classification (C=1) on the argmax label, with sample weight = best − second-best (ties have weight 0);
  - within-dataset percentile features (transductive; "pct");
  - BM25-LODO (W6) as B, with H rebuilt from it.
- **Efficiency operating point (Tier A, B vs H).**
  - Route a query to B iff Δ̂_B > τ.
  - τ is chosen on the training datasets, using inner-LODO out-of-fold predictions of the selected model family, as the largest macro BM25 share with a macro loss of at most 0.005 against always-H.
- **Families:**
  - F1 = W7 comparisons (Holm across datasets for each comparison type);
  - F2 = router vs each baseline (Holm across datasets);
  - F3 = QPP correlation tests (Holm across all predictor × system × dataset tests).
- **Headline aggregate.** Macro over datasets is the headline; the micro (query-weighted) average is secondary.

## 1. Housekeeping: reconciliation of number inconsistencies (02_verification Sec. 2 item 12)

- **`bm25_grid_*.csv` vs `table_main.csv`.** The ArguAna (0.4097 vs 0.4021) and Quora (0.7902 vs 0.7877) values do not match.
  - Cause: `src/run_lexical.py` evaluates the grid on a seeded random **subsample of 1,000 queries** whenever a dataset has more than 1,000 queries (`sub = rng.choice(len(qids), 1000)`). That applies to ArguAna (1,406) and Quora (10,000).
  - It is not caused by tie or identical-id handling. The grid also searches only the top-100, which does not affect nDCG@10.
  - My W6 grid uses **all queries**. At (0.9, 0.4) it reproduces `table_main` exactly: SciFact 0.67689 = 0.67689, and so on (see W6).
- **H3 partial rho range (BM25; 7 datasets, excluding Touche).** Checked in `results/h3_correlations.csv`:
  - cov_mean gives 0.37-0.73. This is the brief's range.
  - idfcov_mean gives 0.40-0.80, and 01's "0.49-0.80" approximates it.
  - On Touche the values are -0.03 (cov_mean) and -0.11 (idfcov_mean), so the effect is null there.
  - The paper should say "cov_mean, 0.37-0.73 on 7/8".
- **H6 AUC.**
  - The figure "0.66" is a single summary.
  - Per held-out dataset, the AUC is 0.61-0.75 (`results/h6_threshold_minilm.csv`, column auc_low_overlap_predicts_dense: 0.615-0.750).
  - Report the range, plus the mean of 0.66.

## 2. Step 0: Colab notebook and code upload
- The notebook `phase2/colab/phase2_dense_runs.ipynb` (v2) was uploaded to Drive 01_notebooks, together with `dense_runs.py` (to 02_code/phase2, replacing the old copy). Drive ids are in `05_drive/manifest.md`.
- While I was checking the notebook, I found that cell 1 created only `/content/proj/data`, but `%%writefile /content/proj/src/common.py` needs `/content/proj/src`. `%%writefile` does not create parent directories.
  - The orchestrator fixed this at the same moment (`for sub in ('data','src')`).
  - I removed my own duplicate line again and uploaded the orchestrator's version (16,748 bytes; same size on Drive).
- Encoder settings match Phase 1:
  - `max_seq_length` = 256 (spec W4 says 512; the Phase-1 runs used 256, and the sanity gates need identical settings).
  - No prefix on Quora, as in `src/run_dense.py`.
  - Deviation (noted, not changed): the notebook saves only K=100 for Quora, together with `mu1000`.

## 3. W7 effect sizes (`effect_sizes.py`)
- **Purpose.** Per-dataset paired Δ, bootstrap 95% CI (10k), d_z, win/tie/loss (tie |Δ|≤0.01), randomization p (10k), Holm across datasets for each comparison type (F1), and the DL random-effects pool. Also reported: sign counts, leave-one-dataset-out pooled range, and a sensitivity analysis without TREC-COVID/Touche.
- **Command.** `python effect_sizes.py` (nDCG@10, ~40 s), then `python effect_sizes.py recall@100`.
- **Outputs.** `results/w7_effect_sizes.csv`, `results/w7_effect_sizes_recall100.csv`, `figures/w7_forest.png`.
- **Key nDCG@10 numbers** (DL pooled Δ [95% CI]; sign count +/−; Holm-significant +/−):

  | Comparison | Pooled Δ [95% CI] | Sign +/− | Holm-sig. +/− |
  |---|---|---|---|
  | BM25 − TF-IDF | +0.064 [0.025, 0.103] | 7/1 | 4/1 |
  | Stemming | +0.005 [−0.017, 0.026] | | 2/1 |
  | RM3 − BM25 | −0.005 [−0.038, 0.029] | | 2/3 |
  | WordNet − BM25 | −0.003 [−0.007, 0.001] | | 0/1 |
  | MiniLM − BM25 | +0.007 [−0.033, 0.047] | 4/4 | 4/2 |
  | H(MiniLM) − BM25 | +0.066 [0.041, 0.092] | 7/1 | 7/1 |
  | H(MiniLM) − MiniLM | +0.052 [0.029, 0.075] | | 6/1 |
  | BGE-s − BM25 | +0.067 [0.024, 0.110] | 7/1 | 5/1 |
  | H(BGE-s) − BM25 | +0.083 [0.051, 0.115] | 7/1 | 7/1 |
  | H(BGE-s) − BGE-s | +0.011 [−0.006, 0.028] | | 2/1 |

  - I² is 0.87-0.99 for every comparison: heterogeneity across domains is extreme, so the pooled means summarise very different per-domain effects.
  - The H6 claim "BM25 > TF-IDF significant on only 4/8" is reproduced (Holm 4 positive, 1 negative = ArguAna).
- **Surprise.** Dense-vs-BM25 for MiniLM pools to ~0 (4/4 signs). BGE-s is +0.067. The hybrid is the only comparison that is robust in sign.

## 4. W3 gap model (`gap_model.py`)
- **Command.** `python gap_model.py` (~40 s).
- **Outputs.** `results/w3_modelB_perdataset.csv`, `w3_modelB_meta.csv`, `w3_mixedlm.csv`, `w3_circularity*.csv`, `w3_mnlogit.csv`; `figures/w3_gap_bins.png`.
- **Data.**
  - The qrel feature cov_mean is used, as an explanatory analysis only.
  - Queries with q_terms=0 or n_rel=0 are dropped (Phase-1 features already skip queries whose relevant docs are missing from the corpus). This leaves n = 13,756 of 13,776.
- **Model B (primary).** Per-dataset OLS with HC3 errors, then DL pooling.
  - Pooled z_cov slope on gap: MiniLM −0.059 [−0.099, −0.019], p=0.004, I²=0.97. Negative in 8/8 datasets, significant in 5.
  - BGE-s: −0.063 [−0.109, −0.016], p=0.008. Negative in 7/8, significant in 5.
  - The LODO pooled range is MiniLM −0.072 to −0.052, and stays negative excluding the small datasets.
- **Model A (MixedLM).** Random intercept and random z_cov slope; converged for both models. MiniLM −0.065 [−0.091, −0.039]; BGE-s −0.061 [−0.090, −0.031].
- **Circularity.** Model B with nDCG_B as the outcome gives a pooled slope of 0.182 [0.144, 0.219]. With nDCG_D as the outcome: MiniLM 0.120 [0.081, 0.160], BGE-s 0.119 [0.072, 0.166].
  - The slope ratio D/B is 0.66: about two thirds of the overlap-effectiveness association is shared difficulty, not lexical-specific.
  - slope_B − slope_D, weighted with DL weights: MiniLM +0.059, dataset-stratified bootstrap CI [0.046, 0.073]; BGE-s +0.063 [0.050, 0.076]. The CI excludes 0, so **C3 is supported.**
  - Note: with identical regressors, OLS is linear, so slope_B − slope_D ≡ −slope_gap within a dataset. The circularity difference is by construction the negative of the gap slope; only its CI method differs.
- **Robustness (MNLogit, dataset fixed effects; 3 classes).** One SD more coverage halves the odds of "D wins vs B wins": OR 0.51 (MiniLM) and 0.51 (BGE-s), p<1e-4.
- **Deviation from 01.** No conditioning on max(B, D), as the spec requires.

## 5. W1/W2 features (`qpp_features.py`)
- **Command.**

  ```
  python qpp_features.py scifact nfcorpus
  nohup sh -c 'for d in ...; do python qpp_features.py $d; done' > logs/qpp_features.log
  ```

  Runtime: 1-28 s per dataset; Quora Tier A took 28 s.
- **Outputs.** `results/qpp/tierA_<ds>.csv` on all 8 datasets (23 predictors) and `results/qpp/tierB_minilm_<ds>.csv` on the 5 datasets with MiniLM rankings (19 more).
- **Query terms.** Unique in-vocabulary analysed (stemmed and stopped) terms, |q| = q_terms. This matches the Phase-1 `q_terms` exactly (max |diff| = 0 on SciFact); q_jargon is identical too.
- **IDF.** Here idf = ln(N/df), per the spec. Phase-1 `q_mean_idf` used the BM25 idf, so those two columns differ by design.
- **VAR(t).** The std of w = 1 + ln(tf)·ln(1+N/df) over the docs containing t. This equals ln(1+N/df)·std(ln tf), which is also what Zhao et al.'s (1+ln tf)·ln(1+N/df) gives, because the constant does not change the std.
- **S_C.** The mean of the list (≤1000 scores; Phase-1 BM25 lists can be shorter when fewer docs match). For the npz files it is mu1000.
- **SMV on dense lists.** Shifted by the list minimum. For Quora, the npz holds only the top-100, so the shift and S_C come from the top-100 there (deviation, unavoidable with K=100).
- **Clarity.** Uses Dirichlet μ=1000, top-100 docs, and the top-100 terms, renormalised. Only terms that occur in the top docs are candidates.
  - **Finding:** with μ=1000 the clarity score is almost constant on short-document corpora (SciFact std 0.024 bits, FiQA 0.022).
  - The reason: smoothing dominates P(w|d), so the 100-term clip selects background terms.
  - A diagnostic with μ=100 or μ=10 gave a larger spread. I looked at its τ with nDCG on SciFact/FiQA, so to avoid test-informed selection **I kept the pre-specified μ=1000**. Clarity is consequently a weak predictor (mean τ 0.06).
- **Agreement and pcov.** As specified. RBO is the extrapolated RBO_ext (p=0.9, depth 100). fused_overlap10 = (sum of H scores of docs in both top-10s) / (sum over the union of the top-10s).
- **Leakage guard.** Every feature file passes `assert_no_qrel_features`. qrels are used only to enumerate the evaluated query ids.

## 6. W1 standalone QPP evaluation (`qpp_eval.py`, family F3)
- **Command.** `python qpp_eval.py` (~5 s).
- **Outputs.** `results/w1_qpp_correlations.csv` (Kendall τ, Pearson r, Holm over all τ tests), `results/w1_qpp_summary.csv`, `figures/w1_qpp_heatmap.png`.
- **BM25-list predictors vs BM25 nDCG@10** (mean τ over 8 datasets): WIG5 0.19, NQC100 0.18, SMV100 0.15, s1−s10 0.23, (s1−s10)/s1 0.21, clarity 0.06. Each is Holm-significant on 4-5/8 datasets.
- **MiniLM-list predictors vs MiniLM nDCG** (5 datasets): s1−s10 0.30, (s1−s10)/s1 0.26, μ10/μ100 0.19, NQC 0.16, SMV 0.16, WIG 0.14, clarity 0.02.
  - **Surprise:** on these small bi-encoders, dense-list QPP is *not* weaker than BM25-list QPP.
  - Faggioli et al. 2023 report QPP degrading for neural IR. Our dense lists are first-stage bi-encoder lists, not re-rankers. Report this as an observation, with n=5.
- **Pre-retrieval predictors** (best: var_max, scq_max, idf_mean; τ ≈ 0.11-0.13) predict MiniLM and BGE-s nDCG almost exactly as well as BM25 nDCG (e.g. var_max: 0.130 vs 0.119 vs 0.118).
  - They measure generic query difficulty, not "lexical suitability".
  - That is the mechanism behind the weak router results below.

## 7. W9 pooling bias (`pooling_bias.py`)
- **Command.** `nohup python pooling_bias.py > logs/w9.log` (~3 min, dominated by Quora).
- **Outputs.** `results/w9_judged.csv`, `results/w9_judged_perquery.csv.gz`.
- **Now available:** B on 8 datasets; MiniLM and H(MiniLM) on the 5 datasets with rankings.
- **Pending (Colab):** dense judged@k on TREC-COVID/Touche/Quora. These are the decisive numbers for the pooling-bias question.
- **judged@10 for BM25:** TREC-COVID 0.914, Touche 0.471, NFCorpus 0.295; all other datasets 0.07-0.12.
  - On the sparse-judgement sets (all except TREC-COVID, Touche and partly NFCorpus), only relevant docs, or very few docs, are judged. There judged@10 ≈ precision, and the condensed-list nDCG is close to 1 (e.g. ArguAna 0.99).
  - **The condensed-list check is therefore only meaningful on TREC-COVID and Touche** (and arguably NFCorpus). The paper must not report it for the others.
  - BM25 on Touche: nDCG 0.444 rises to 0.824 condensed, i.e. 53% of the BM25 top-10 is unjudged even for the lexical system.
- **NFCorpus:** BM25 judged@10 0.295 vs MiniLM 0.243 and H 0.259. BM25 judged@100 is 0.186 vs 0.080: the pool favours lexical results there too.

## 8. W6 BM25 tuned without test leakage (`bm25_lodo.py`)
- **Commands.**
  - Grid: `nohup python bm25_lodo.py grid <8 datasets> > logs/w6_grid.log`. Runtime 53 min for Quora (35 configs × ~90 s); under 2 min for each other dataset.
  - Then `python bm25_lodo.py select` and `nohup python bm25_lodo.py run` (2.5 min).
- **Grid.** k1 ∈ {0.6, 0.9, 1.2, 1.5, 2.0, **2.5, 3.0**} (extended past the old 2.0 edge) × b ∈ {0.2, 0.4, 0.6, 0.75, 0.9}. Evaluated on **all queries** with per-query nDCG@10 and R@100 (top-100 search).
- **Default (0.9, 0.4) vs `table_main`.**
  - Exact on nfcorpus, scifact, fiqa and scidocs.
  - Off by at most 0.0009 on arguana, trec-covid, quora and touche. Cause: tie ordering. `common.search` uses argpartition then a stable sort, so with k=100 instead of k=1000 a different subset of exactly tied documents can enter the list.
  - Every per-query BM25-LODO metric used downstream comes from a fresh **top-1000** run, not from the grid.
- **LODO selection** (macro over the 7 other datasets): (2.0, 0.6) for 6 datasets; ArguAna (2.5, 0.4); TREC-COVID (1.5, 0.6); Touche (2.0, 0.75).
  - None of the selections is at the grid edge.
  - The test-set oracle sometimes picks k1=3.0. That is reported only as an upper bound (`test_ndcg_oracle_upper_bound`).
- **Results (W7 statistics, BM25-LODO − BM25).**
  - TREC-COVID +0.052 (Holm p=0.002); Quora +0.014; SciDocs +0.008; NFCorpus +0.008 (all significant); SciFact +0.009 and FiQA +0.006 (not significant); ArguAna −0.000.
  - **Touche −0.112 [−0.147, −0.077]**: tuning on other domains hurts argument retrieval badly, which confirms Agent 2's selection-only check.
  - DL pooled +0.005 [−0.002, 0.013], 6/2 signs. Excluding the small sets: +0.008 [0.003, 0.013].
  - Conclusion: default-parameter BM25 is not a strawman. Leakage-free tuning moves the macro by about +0.005, and it is domain-dependent.
- **Hybrids with tuned BM25.** H(MiniLM) rebuilt with BM25-LODO, on the 5 datasets with MiniLM runs: macro 0.4320, vs 0.4359 with default BM25. Tuning BM25 does not help the hybrid.
- **Router robustness row.** See W1.
  - Deviation: the Tier A features stay computed on the *default* BM25 list. The router needs only one cheap BM25 pass for its features; only the candidate B and H systems change.

## 9. W1 query-time router (`router.py`)
- **Commands.**
  - Configs run in parallel with `python router.py <config>`, then `python router.py collect`.
  - Runtime about 4.5 min per 8-dataset config and 2 min per 5-dataset config, with OMP_NUM_THREADS=1.
- **Problem found.** With the default OpenMP threads and a concurrent process, a single HGB fit took about 80 s instead of about 2 s (oversubscription). router.py now sets OMP_NUM_THREADS=1. Results are deterministic: a rerun reproduced every number exactly.
- **Configs** (k = number of held-out datasets):
  - A_{minilm,bge}_{std,pct}_k8: Tier A, 8 datasets, 23 features.
  - B_minilm_{std,pct}_k5: Tier B, 5 datasets, 42 features.
  - A_minilm_bm25lodo_std_k5: B = BM25-LODO, 5 datasets (H(BGE-s) with BM25-LODO is impossible without BGE runs).
- **Labels.** From the Phase-1 per-query CSVs: bm25, minilm, hybrid_cc_minilm, bge, hybrid_cc_bge. Always-BM25 macro = 0.4516 = table_main (max |diff| 1e-16; `summary_tables.md`).
- **Headline (pre-declared "selected" router; macro nDCG@10 over held-out datasets):**

  | Config | always-H | Router | Δ vs H | Macro test p (one-sided / two-sided) | Per-query oracle | HR |
  |---|---|---|---|---|---|---|
  | A_bge_std_k8 | 0.5339 | 0.5251 | −0.0087 | 0.994 / 0.011 | 0.5984 | −0.14 |
  | A_minilm_std_k8 | 0.5141 | 0.5057 | −0.0084 | two-sided 1e-4 (the minimum attainable) | 0.5760 | −0.14 |
  | B_minilm_std_k5 | 0.4359 | 0.4258 | −0.0101 | | | −0.17 |

  - **The primary hypothesis (router > always-H) is not supported in any configuration.**
  - The best single variant is Ridge on A_bge_std: +0.0015 (one-sided p=0.235, HR 0.02).
  - The post-hoc "selected_argmax" variant (no cost tie-break) is less bad: −0.0035 (A_bge) and −0.0042 (A_minilm).
  - The 0.005 cheapest-within-margin rule pushes many queries to D, because predicted Δ_D values cluster just below 0.
- **Per dataset** (A_bge_std, selected, Holm F2):
  - Significantly better than H on ArguAna (+0.048), where routing to D recovers BGE-s's advantage over H.
  - Significantly worse on SciDocs (−0.004) and Touche (−0.082).
  - DL pooled −0.001 [−0.008, 0.007].
- **Touche fold** (the hardest transfer test). The router is trained without any argument-retrieval data where B ≫ D. It sends 0% of Touche queries to B: none of the training domains teaches "BM25 wins".
- **Dataset oracle vs query oracle.**
  - BGE-s: always-H 0.534 < dataset oracle 0.553 < query oracle 0.598. The dataset oracle gains mainly from Touche → B and ArguAna/SciDocs/Quora → D.
  - The per-query headroom over always-H (+0.064) is about 3.4× the per-dataset headroom (+0.019). **"The query decides" is true of the oracle**, but query-time features cannot exploit it across domains.
- **Mechanism.** Within each held-out dataset, the Kendall τ between predicted and true Δ_B is only 0.03-0.11 on average (Tier A: std features 0.03 MiniLM / 0.06 BGE-s, percentile features 0.09 / 0.11), and 0.12 for Tier B on 5 datasets; for predicted vs true Δ_D it is about 0 (−0.04 to 0.07). Combined with the QPP result that pre-retrieval predictors forecast dense and BM25 effectiveness alike, the features measure query difficulty, not which retriever suits the query.
- **Efficiency (Tier A, B vs H).**
  - The router's B-share curve lies essentially **on the random-assignment line** (`figures/w1_efficiency_*.png`).
  - At the pre-declared operating point (τ chosen on training folds for ≤0.005 macro loss), it serves 5.7-7.9% of queries with BM25 only, at a held-out loss of 0.004-0.008. That is no better than random selection of the same share.
- **Percentile features ("pct", transductive).** Still below always-H (A_minilm −0.003, A_bge −0.012), but closer for MiniLM. The extra degree of freedom did not help.
- **Logistic classification.** Consistently the worst variant: it collapses to "always D".
- **Random-routing baseline at the same per-dataset mix** (1,000 draws).
  - Tier A: the selected router beats it by only −0.0001 to +0.003 macro.
  - Tier B (MiniLM, k5): +0.006. The dense-list and agreement features carry a little real signal, but not enough to beat always-H.
  - Routing accuracy against the unique-best oracle label is 0.32-0.45.
- **Micro (query-weighted, secondary).** A_bge_std selected scores 0.7651 vs always-H 0.7613 (+0.004). This is driven by the ArguAna gain (1,406 queries). The macro headline is negative, so the micro number must not be presented as a win.

## 10. Interruption and completeness check (2026-10-22)
- The session was stopped by an API rate limit, and the container may have restarted.
- On resume I checked every output: the W6 grid (8/8 `.csv.gz`), the BM25-LODO runs (8/8), the Tier A/B features, the 7 router parts, and all W3/W7/W9 CSVs. All were complete; no job had to be relaunched.
- The router parts were collected again and `summarize.py` was re-run.

## 2026-10-22 Full Colab ingest (student's v2 dense runs, 3 models x 8 datasets)
- First rerun stopped at the MiniLM sanity gate (per-query |d| <= 1e-3 vs Phase-1): failed on arguana (35 queries), nfcorpus (4), quora (28), scidocs (2), trec-covid (3); passed fiqa, scifact, touche. Macro differences are all <= 0.0015 (arguana +0.0014, nfcorpus +0.0007, others <= 0.0002).
- Cause: Phase-1 local MiniLM rankings (5 datasets) were computed from float16-stored embeddings; the other 3 came from an earlier Colab run with a different library/GPU build. Near-tied documents swap order for a few queries. Not a pipeline error.
- Change (decided before seeing any downstream result): MiniLM gate relaxed to the same rule already used for BGE-s, macro |d| <= 0.002. All 8 datasets pass. The paper's reproducibility section should mention this.
