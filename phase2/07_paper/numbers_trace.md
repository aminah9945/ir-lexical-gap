# numbers_trace.md: where every number in `main.tex` comes from

**Path abbreviations.**
- `P1/` = `results/` (Phase 1, read-only).
- `A3/` = `phase2/03_experiments/results/`.
- `A4/` = `phase2/04_improvements/results/`.
- `LOG3` = `phase2/03_experiments/EXPERIMENT_LOG.md`.
- `INT` = `phase2/04_interpretation.md`.
- `BPC` = `phase2/06_base_paper_comparison.md`.
- `VER` = `phase2/02_verification.md`.

**Conventions.**
- All values are nDCG@10 unless noted.
- Rounding is to the precision shown in the paper; "≈" marks rounding.
- Table numbers follow the compiled PDF (revised version): I `tab:data`, II `tab:main`, III `tab:effects`, IV `tab:router`, V `tab:improve`, VI `tab:sens`. Figures: Fig. 1 `fig:indomain`, Fig. 2 `fig:fewshot` (the earlier win-share and router-bar figures were removed in revision).
- `EX` = `phase2/07_paper/analysis_extra/extra_numbers.json`, produced by `analysis_extra/extra_numbers.py` (post-review, exploratory except the micro average). Bootstrap = stratified percentile, 10^4 resamples, seed 42; p = `common2.perm_test_macro`.
- In `w7_effect_sizes.csv`, `summary4.md` and `w1_router_tests.csv`, `MACRO` / `POOLED_DL` are the rows named so in the `dataset` column.

## Abstract and Introduction

| Number in paper | Meaning | Source |
|---|---|---|
| eight collections | the datasets | `P1/table_main.csv` (8 datasets) |
| ρ = +0.33, p = 0.42, n = 8 | corpus jargon vs BM25 nDCG (H2) | `P1/verdicts.json` → H2.corpus_jargon (rho 0.3333, p 0.4198) |
| ρ = −0.60, p = 0.12 | jargon vs MiniLM−BM25 gain | `P1/verdicts.json` → H2.corpus_jargon_vs_dense_gap (rho −0.5952, p 0.1195) |
| +0.064 | per-query-oracle headroom over always-H (BGE-s) | `A3/w1_router_macro.csv` A_bge_std_k8: oracle_query 0.5984 − always_H 0.5339 = 0.0645 |
| +0.019 | dataset-oracle headroom | same row: oracle_dataset 0.5526 − 0.5339 = 0.0187 |
| 3.4 times | ratio of the two | unrounded 0.06452 / 0.01877 = 3.44 → 3.4 (corrected note, critique COULD-8) |
| 0.525 vs 0.534 | selected router vs always-H, Tier A BGE-s | `A3/w1_router_macro.csv` A_bge_std_k8 / selected: router 0.5251, always_H 0.5339 |
| +0.010 to +0.014 (10–20 labels) | few-shot collection-level selection, BGE-s | `A4/summary4.md` few-shot table: bge k=10 0.0103, k=20 0.0142 (raw: `A4/e4_fewshot_dataset_selection.csv`) |
| seven configurations | number of router configs | `A3/w1_router_macro.csv` (7 configs) |
| within 0.005, all eight | BM25 vs Pyserini flat | `P1/table_main.csv` (bm25) vs `report/narrative.py` PYSERINI_BM25; max \|Δ\| 0.005 (ArguAna) |
| 8k training queries | Quora, 10,000 queries × 4/5 folds | `A4/e4_indomain_perdataset.csv` (quora n = 10000); 5-fold CV |

## Related work (numbers about other papers, read in primary sources by Agent 6)

| Number | Source |
|---|---|
| BEIR 18 datasets; k1 = 0.9, b = 0.4 multifield; BM25+CE beats BM25 on 16/18, fails on ArguAna and Touché | `BPC` §2.1 [A6-read, arXiv 2104.08663] |
| "up to 0.52" (Touché denoising) | `BPC` §2.15 [A6-read, arXiv 2407.07790] |
| Faggioli: 19 QPP methods, 7 BoW + 7 BERT systems, DL'19 and Robust04 | `BPC` §2.11 |

## Setup

| Number | Source |
|---|---|
| Table I: Docs, \|q\|, rel/q, jargon | `P1/domain_stats.csv`: n_docs, avg_q_len, avg_rel, corpus_jargon (rounded) |
| Table I: Queries (ArguAna 1,406) | `A3/w1_router_perdataset.csv`, column n (evaluated test queries; `domain_stats` lists 1,401 for ArguAna because 5 queries were skipped for the features) |
| Zipf < 3 | Phase-1 jargon definition (brief; `src/features.py`) |
| k1 = 0.9, b = 0.4 | `src/common.py`; brief |
| MiniLM 22M, BGE-s 33M parameters | brief ("22-33M params"); model cards |
| max_seq_length 256 | `LOG3` §2; `BPC` §2.18 |
| top-1000, α = 0.5; RRF k = 60 | `BPC` "Systems"; §2.14 |
| CE max_length 512, top-100 | `INT` E3; `phase2/04_improvements/RESULTS.md` |
| 23 Tier A features (15 pre + 8 post) | header of `A3/qpp/tierA_scifact.csv` |
| 19 Tier B features (8 + 11) | header of `A3/qpp/tierB_minilm_scifact.csv` |
| Tier B on 5 datasets | `A3/qpp/` (tierB_minilm_* for 5 datasets) |
| cost tie-break 0.005 | `LOG3` §0 |
| 10^4 sign flips / bootstrap resamples; seed 42 | `LOG3` header |
| decision rule: Holm p < 0.05 and Δ ≥ +0.005 | `INT` "Common rules" |

## Results A: baselines and domain hypotheses

| Number | Source |
|---|---|
| Table II `tab:main`, all system columns | `P1/table_main.csv` ndcg@10 for bm25, tfidf, minilm, bge, hybrid_cc_minilm, hybrid_cc_bge (rounded to 3 decimals) |
| Table II, Ref column | `report/narrative.py` PYSERINI_BM25 (= Kamalloo et al. flat column, verified in `BPC` §2.2) |
| Table II, macro row | means of `P1/table_main.csv`: bm25 0.4516, tfidf 0.3675, minilm 0.4457, bge 0.5147, hybrid_cc_minilm 0.5141, hybrid_cc_bge 0.5339; Ref mean 0.4511 |
| mean \|Δ\| = 0.0018 | `BPC` §2.2 |
| BGE-s within 0.02 of model card on 8/8 | `BPC` §2.18 |
| BM25 > TF-IDF Holm-significant on 4/8 | `A3/w7_effect_sizes.csv` BM25 − TF-IDF, sig_pos_holm 4 (also `P1/verdicts.json` H1 holds 4) |
| ArguAna 0.534 vs 0.402 | `P1/table_main.csv` arguana tfidf 0.5343, bm25 0.4021 |
| 0.528 with (k1, b) = (2.5, 0.9) | `A3/summary_tables.md` W6 table, arguana oracle_k1 2.5, oracle_b 0.9, test_ndcg_oracle_upper_bound 0.5280 |
| +0.064 [0.025, 0.103] | `A3/w7_effect_sizes.csv` BM25 − TF-IDF (ci_hi 0.102 in the machine table; 0.103 in `RESULTS_SUMMARY.md` §2, rounding of 0.1025) |
| −0.005 [−0.039, 0.029] without TREC-COVID and Touché | `A3/summary_tables.md` sensitivity table, BM25 − TF-IDF |
| Boolean AND empty for 62–93% on 6/8 | `P1/boolean_set_summary.csv` empty_frac: scifact 0.933, fiqa 0.684, scidocs 0.864, arguana 0.920, trec-covid 0.620, quora 0.695 (nfcorpus 0.396 and touche 0.061 are outside) |
| ρ +0.33 / −0.60, p 0.42 / 0.12 | `P1/verdicts.json` H2 (see above) |
| MDE \|ρ\| ≈ 0.85 (α = .05, power .8) | `VER` W5 |
| FiQA MiniLM gain +0.132; FiQA jargon 0.034 | `P1/h2_domain_table.csv` gap_minilm 0.1318; corpus_jargon 0.0341 |
| I² = 0.87–0.99 | `A3/summary_tables.md` W7 I2 column (0.875–0.991) |
| BM25-LODO +0.005; Touché −0.112 [−0.147, −0.077] | `A3/w7_effect_sizes.csv` BM25-LODO − BM25; `LOG3` §8 (Touché) |
| H(BGE-s) 7/8 Holm-significant; recall@100 better on 8/8 | `A3/w7_effect_sizes.csv`; `A3/w7_effect_sizes_recall100.csv` (`RESULTS_SUMMARY.md` §2) |
| CC 0.534 vs RRF 0.528 (BGE-s) | macro of `P1/table_main.csv` hybrid_cc_bge 0.5339, hybrid_rrf_bge 0.5275 |
| Table III `tab:effects` (all rows) | `A3/summary_tables.md` W7 pooled table (delta, ci_lo, ci_hi, sign_pos/neg, sig_pos/neg_holm); CI bounds as in `RESULTS_SUMMARY.md` §2 |

## Results B: overlap and QPP

| Number | Source |
|---|---|
| partial ρ 0.37–0.73 on 7/8; Touché null | `P1/h3_correlations.csv` cov_mean; `LOG3` §1 |
| 0.182 [0.144, 0.219]; 0.120; 0.119; ratio 0.66 | `A3/w3_circularity.csv` (pooled_slope_B 0.1815 [0.1439, 0.2190]; pooled_slope_D 0.1204 / 0.1187; ratio 0.663 / 0.654) |
| +0.059 [0.046, 0.073]; +0.063 [0.050, 0.076] | `A3/w3_circularity.csv` diff_weighted, diff_boot_lo/hi |
| gap slope −0.059 [−0.099, −0.019], 8/8 negative | `A3/w3_modelB_meta.csv` minilm (−0.0588 [−0.0991, −0.0185], sign_neg 8) |
| mixed model −0.065 | `A3/w3_mixedlm.csv` minilm −0.0647 |
| proxy ρ 0.39–0.53 | `A3/w2_proxy_rho.csv` pcov_D10 row (0.39–0.53) |
| proxy gap slope −0.006 [−0.022, 0.011] | `A3/w2_proxy_modelB_meta.csv` pcov_D10 (−0.0055 [−0.0218, 0.0109]) |
| VAR_max τ 0.130 / 0.119 / 0.118 | `A3/w1_qpp_summary.csv` rows pre/bm25, pre/minilm, pre/bge var_max mean_tau |

## Results C: router (Table IV `tab:router` and text)

| Number | Source |
|---|---|
| Table IV, all cells | `A3/summary_tables.md` W1 macro table, variant = selected (configs A_bge_std_k8, A_minilm_std_k8, B_minilm_std_k5, A_minilm_bm25lodo_std_k5): B, D, H, oracle_ds, oracle_q, router, minus_H, p1_vs_H, pctB/D/H; Holm ± from the "sig_better / sig_worse" table |
| best variant +0.0015, p = 0.24 | `A3/w1_router_tests.csv` A_bge_std_k8 ridge MACRO: delta 0.00147, p_one_sided 0.235 |
| ArguAna +0.048, 91% to D; SciDocs −0.004; Touché −0.082, 0% to B | `A3/w1_router_tests.csv` (A_bge_std_k8 selected); `A3/summary_tables.md` per-dataset A_bge_std_k8 (pct_D 0.913; touche pct_B 0.000) |
| τ(pred, true Δ_B) 0.03–0.11 (Tier A), 0.12 (Tier B); Δ_D ≈ 0 | `LOG3` §9 "Mechanism" |
| vs random same mix −0.0001 to +0.008 | `A3/w1_router_macro.csv` selected rows: router − random_same_mix (−0.0001 … +0.0081) |
| 5.7–7.9% BM25-only, loss 0.003–0.008 | `A3/summary_tables.md` "Efficiency operating point" table (macro of `A3/w1_efficiency_operating.csv`), Tier A 8-dataset configs: share_B 0.0567–0.0786; loss_vs_H 0.0033–0.0076 |

## Results D: improvement attempts (Table V `tab:improve` and text)

| Number | Source |
|---|---|
| Table V Δ and p_Holm | `A4/f4_primary.csv` |
| P1 FiQA −0.026, Holm p = 0.013 | `A4/e1_tests_minilm.csv` e1_ridge fiqa (−0.0257, p_holm 0.0125) |
| HGB +0.0027 (p = 0.11) | `A4/e1_tests_minilm.csv` e1_hgb MACRO (0.0027, p_one_sided 0.107) |
| LODO-global +0.0037 (p = 0.03; DL CI includes 0) | `A4/e1_tests_minilm.csv` e1_global MACRO 0.0037, p_one_sided 0.0299; POOLED_DL [−0.0042, 0.0139] |
| best α per collection +0.012; per-query α oracle +0.069 | `A4/e1_perdataset_minilm.csv` MACRO: oracle_ds 0.4482 − H 0.4359 = 0.0123; oracle_q 0.5052 − 0.4359 = 0.0693 |
| 150 tuning queries; α̂ = 0.5 on 3/5; SciDocs 0.7 vs 0.2; r = −0.49; −0.0044 | `A4/e2_alpha_perdataset.csv`; `A4/e2_tests.csv` |
| CE(H) −0.040; ArguAna −0.120 | `A4/f4_primary.csv` P3 (−0.0397); `A4/e3_ce_perdataset.csv` arguana hybrid_cc_minilm delta −0.1204 |
| CE(BM25) +0.025, p = 0.004, Holm +4/−1 | `A4/e3_ce_tests.csv` bm25 CE(bm25)-base MACRO (0.0249, p_one_sided 0.0036, 4/1) |
| TREC-COVID +0.143; Touché −0.164 [−0.230, −0.100] | `A4/e3_ce_perdataset.csv` (0.1426, −0.1641); CI from `A4/e3_ce_tests.csv` row bm25 / CE(bm25)-base / webis-touche2020 (ci −0.2295, −0.0997) |
| CE(BM25) − H(BGE-s) −0.059, 5/8 Holm-negative | `A4/e3_ce_tests.csv` CE(bm25)-ref_hybrid_cc_bge MACRO (−0.0590, sig_neg 5) |
| ≤150 queries; 50/49 for TREC-COVID/Touché | `A4/e3_ce_perdataset.csv` column n |
| P4 +0.0169, Holm p 4e-4 | `A4/f4_primary.csv` |
| router − fixed −0.0005, p = 0.66 | `A4/e4_indomain_tests.csv` A_bge_std_k8 router-fixed MACRO (−0.0005, p_one_sided 0.657) |
| MiniLM Tier A −0.0036; Tier B −0.0016 | `A4/e4_indomain_tests.csv` router-fixed MACRO for A_minilm_std_k8, B_minilm_std_k5 |
| prior: D for ArguAna/SciDocs, B for Touché | `A4/e4_indomain_perdataset.csv` A_bge_std_k8: indomain_best_fixed equals always_D (arguana, scidocs) and always_B (touche) |
| Quora +0.0046 [0.0026, 0.0066], Holm p 0.0008 | `A4/e4_indomain_tests.csv` A_bge_std_k8 router-fixed quora (0.00463 [0.00264, 0.00659], p_holm 0.0008) |
| (BGE-s) gain > 0.2: 2–27% of queries, 23–87% of headroom | `A4/e4_headroom_concentration.csv` model = bge: share_queries_gt_02 0.020–0.267; share_from_gt_02 0.228–0.875 |
| ≤2 relevant docs carry 83–100% (ArguAna, SciFact, Quora) | `A4/e4_label_consistency.csv` headroom_share_nrel_le2 (0.827–1.000) |

## Results E: collection-level selection (exploratory)

| Number | Source |
|---|---|
| split-half +0.0182 (in-sample +0.0188), MiniLM +0.0130 (+0.0134) | `A4/e4_splithalf.csv` means over datasets of splithalf_gain_mean and oracle_ds_gain_insample (`A4/RESULTS.md` E4) |
| k = 5/10/20/30: BGE-s +0.004/+0.010/+0.014/+0.016; MiniLM −0.004/+0.005/+0.009/+0.010 | `A4/summary4.md` few-shot table (0.0040, 0.0103, 0.0142, 0.0157; −0.0043, 0.0045, 0.0085, 0.0100) |
| 1,000 splits/draws | `phase2/04_improvements/e4_why_routing_fails.py` `descriptive()` |

## Discussion and limitations

| Number | Source |
|---|---|
| τ ≤ 0.12 | as in Results C |
| Touché −0.112 | as in Results A |
| judged@10 0.471; 0.444 → 0.824; NFCorpus judged@100 0.186 vs 0.080 | `A3/w9_judged.csv` (also `A3/summary_tables.md` W9) |
| 49 and 50 queries | `P1/domain_stats.csv` n_queries |
| flat vs multifield gap up to 0.077 | `BPC` §2.2 table: Touché 0.444 (ours) vs 0.367 (multifield) |
| scikit-learn 1.9.1, statsmodels 0.15.0, SciPy 1.18.1 | `LOG3` header |

## Added in revision (critique 08), source `EX` unless stated

| Number in paper | Source |
|---|---|
| Table III "Macro" column (+0.084, +0.000, −0.004, −0.004, −0.002, −0.006, +0.063, +0.063, +0.082, +0.019) | `A3/w7_effect_sizes.csv`, rows `POOLED_DL(k=8)`, column `macro_delta` (0.0841, 0.0004, −0.0036, −0.0036, −0.0018, −0.0059, 0.0630, 0.0625, 0.0822, 0.0192) |
| BM25-LODO "+0.005 pooled (−0.002 macro)" | same rows: delta 0.0055, macro_delta −0.0018 |
| recall@100 higher on all 8 (7 Holm-significant) | `A3/w7_effect_sizes_recall100.csv` H(BGE-s) − BM25 POOLED_DL(k=8): sign_pos 8, sig_pos_holm 7 |
| Table IV 95% CI column: [−0.015, −0.003], [−0.012, −0.005], [−0.016, −0.005], [−0.017, −0.007] | `EX` router/<cfg>/selected/all: A_bge_std_k8 [−0.01509, −0.00272]; A_minilm_std_k8 [−0.0120, −0.0052]; B_minilm_std_k5 [−0.01554, −0.00467]; A_minilm_bm25lodo_std_k5 [−0.01713, −0.00654] |
| Router BGE-s without Touché +0.002 [−0.003, +0.006] | `EX` A_bge_std_k8/selected/excl_touche: +0.0017 [−0.0028, +0.0060], p1 0.235 |
| without Touché and TREC-COVID +0.004 [−0.001, +0.008], p = 0.06; abstract "+0.004 (95% CI [−0.001, +0.008])" | `EX` A_bge_std_k8/selected/excl_touche_covid: +0.0036 [−0.0010, +0.0081], p1 0.060 |
| micro 0.765 vs 0.761 (+0.004) | `A3/w1_router_macro.csv` A_bge_std_k8 / selected: router_micro 0.7651, always_H_micro 0.7613 (recomputed in `EX`: 0.7651 / 0.7613) |
| best variant 95% CI [−0.002, +0.005] | `EX` A_bge_std_k8/ridge/all: +0.0015 [−0.0025, +0.0053] |
| 35 variant–configuration pairs | 7 configs × 5 variants (`A3/w1_router_macro.csv`) |
| logistic ≥91% dense in five of seven configurations | `A3/w1_router_macro.csv` logreg pct_D: 0.999, 0.913, 0.999, 0.825, 0.622, 1.000, 0.999 |
| Kendall τ on the same five collections: s1−s10 0.30 vs 0.24; WIG 0.14 vs 0.17; NQC 0.16 vs 0.19 | `A3/w1_qpp_correlations.csv`, mean kendall_tau over nfcorpus, scifact, fiqa, scidocs, arguana; D-list/minilm vs B-list/bm25: s1_s10 0.296 / 0.239; wig5 0.135 / 0.172; nqc100 0.162 / 0.186 |
| Table V CIs: P1 [−0.011, −0.003]; P2 [−0.006, −0.003]; P3 [−0.057, −0.023]; P4 [+0.010, +0.023] | `EX` P1_e1_ridge [−0.0114, −0.0025]; P2_e2_rest [−0.0059, −0.0029] (held-out sets reconstructed exactly, n_rest matches `A4/e2_alpha_perdataset.csv`); P3_ce_hybrid [−0.0569, −0.0227]; P4_router_H [+0.0105, +0.0233] |
| P4 p "< 10^-3" | `A4/f4_primary.csv` p_holm 0.0004 (permutation floor 1/10001 × 4) |
| P4 router − fixed [−0.003, +0.002] | `EX` P4_router_fixed: −0.00046 [−0.0028, +0.0018] |
| CE(BM25) − BM25 [+0.009, +0.041] | `EX` ce_bm25_all: +0.0249 [+0.0089, +0.0411] |
| CE direction matches BEIR on seven, null on ArguAna in both | `BPC` §2.1 table (ArguAna ours −0.000, BEIR −0.004) |
| Noise reference: stemming macro diff 0.0004; up to 0.09 on single collections; oracle +0.043 | `EX` noise_floor["bm25[stop]"]: macro_diff 0.00043, oracle_gain_macro 0.0429; Touché bm25 0.444 vs bm25[stop] 0.532 (`P1/table_main.csv`) = 0.088 |
| BM25 vs BM25+WordNet oracle +0.013 | `EX` noise_floor["bm25_wordnet"]: 0.0126 |
| Table VI row 1 | `EX` A_bge_std_k8/selected: all −0.0087; excl_touche +0.0017; excl_touche_covid +0.0036 |
| Table VI row 2 | `EX` A_minilm_std_k8/selected: −0.0084 [−0.0120, −0.0052]; −0.0096 [−0.0137, −0.0059]; −0.0071 [−0.0099, −0.0044] |
| Table VI row 3 | `EX` P4_router_H +0.0169 [+0.0105, +0.0233]; P4_router_H_excl_touche +0.0069 [+0.0033, +0.0104]; P4_router_H_excl_touche_covid +0.0080 [+0.0039, +0.0121] |
| Table VI row 4 | `EX` ce_bm25_all +0.0249 [+0.0089, +0.0411]; ce_bm25_excl_touche +0.0519 [+0.0361, +0.0679]; ce_bm25_excl_touche_covid +0.0368 [+0.0206, +0.0532] |
| Table VI row 5 (split-half, BGE-s) | `EX` splithalf_all bge 0.0182; splithalf_excl_touche 0.0084; splithalf_excl_touche_covid 0.0099 |
| Table VI rows 6–7 and text "without Touché: −0.006, 0.000, +0.004, +0.006; MiniLM negative for k ≤ 30" | `EX` fewshot: bge k5/10/20/30 excl_touche −0.0061, −0.0002, +0.0039, +0.0057; excl_touche_covid k20 +0.0049; minilm excl_touche −0.0159, −0.0068, −0.0027, −0.0010; minilm k20 excl both −0.0031 |
| "macro gives two 50-query collections a quarter of the weight" | 2/8 collections; `P1/domain_stats.csv` n_queries 50, 49 |
| in-domain tree router constant on TREC-COVID/Touché (40-query folds, leaf 50) | `INT` §4 deviations (HGB min_samples_leaf = 50) |

## Figures (revised)

| Figure | File | Data |
|---|---|---|
| Fig. 1 | `figs/fig_indomain.png` | regenerated by `make_fig_indomain.py` from `A4/e4_indomain_perdataset.csv` and `A3/w1_router_perdataset.csv` (same logic as `summarize4.py`) |
| Fig. 2 | `figs/fig_fewshot.png` | regenerated by `make_fig_fewshot.py` from `A4/e4_fewshot_dataset_selection.csv`, `A4/e4_splithalf.csv`, `A3/w1_router_macro.csv` |

Removed in revision (still on disk, unused): `figs/fig_winshare_bge.png`, `figs/fig_router_bars.png`. The per-collection oracle numbers formerly shown in the win-share figure ("+0.034 to +0.122", "55% Touché") are no longer in the paper.


## 2026-11-03 update (BGE-b integration; supersedes earlier rows where they conflict)

Sources: `A3/` and `A4/` files written by the 2026-10-22 full Colab ingest; `EX2` = `analysis_extra/extra_numbers_v2.json` (same bootstrap/perm procedures as `EX`, recomputed on current per-query files); `P2X` = `analysis_extra/p2_rest_v2.json`.

| Number in paper | Source |
|---|---|
| Table II BGE-b / H(b) columns | `A3/w7_effect_sizes.csv` rows `BGE-b - BM25`, `H(BGE-b) - BM25`, column mean_a (BM25 = mean_b) |
| Table III BGE-b rows | `A3/w7_effect_sizes.csv` POOLED_DL(k=8) rows: delta, ci_lo, ci_hi, sign_pos/neg, sig_pos/neg_holm; macro_delta |
| Touché BGE-b −0.202, H(b) −0.061 | same file, webis-touche2020 rows |
| recall@100 H(b) 8/0, 7 Holm | `A3/w7_effect_sizes_recall100.csv` |
| 18 configurations | `A3/w1_router_macro.csv`, distinct config (18); 90 variant×config pairs |
| Table IV values | `A3/w1_router_macro.csv`, variant=selected (always_*, oracle_*, router, pct_*); Δ CI and p: `EX2` router/<cfg>/selected/all; Holm±: `A3/w1_router_tests.csv` selected vs H, p_holm<0.05 |
| BGE-b headroom +0.065, 3.7×, ds +0.018 | A_bgebase_std_k8: 0.6130−0.5483, 0.5660−0.5483 |
| BGE-b sensitivities +0.003 [−0.002,+0.008], +0.006 [+0.001,+0.010] p 0.01; Tier B +0.007 [+0.002,+0.012] | `EX2` router A_bgebase_std_k8/selected, B_bgebase_std_k8/selected excl_* |
| best pair +0.0053 [+0.002,+0.009] p 0.004; same Ridge default BM25 −0.003; second +0.0015 | `EX2` A_bgebase_bm25lodo_std_k8/ridge, A_bgebase_std_k8/ridge; `w1_router_macro.csv` router_minus_H |
| logreg ≥91% dense in 12/18 | `w1_router_macro.csv` variant=logreg pct_D |
| random-mix gap Tier A −0.0001…+0.003, Tier B +0.002…+0.014 | `w1_router_macro.csv` selected: router − random_same_mix |
| efficiency 3.6–7.9%, loss 0.003–0.008 | `A3/summary_tables.md` efficiency table, Tier A std+pct configs, 8 collections |
| BGE-b per-collection ArguAna +0.052, Quora −0.005, Touché −0.126 | `w1_router_tests.csv` A_bgebase_std_k8 selected vs H, Holm-significant rows |
| P1 −0.0014 [−0.005,+0.003], k=8; FiQA −0.013, SciDocs −0.005, ArguAna +0.015; HGB −0.007; global −0.005; hindsight +0.021 | `EX2` P1_e1_ridge_minilm; `A4/e1_tests_minilm.csv`; `A4/e1_perdataset_minilm.csv` MACRO (oracle_ds − H) |
| P1 secondaries BGE-s +0.0053 [+0.001,+0.010], BGE-b +0.0005 [−0.005,+0.005] | `EX2` P1_e1_ridge_bge, P1_e1_ridge_bgebase |
| P1 interim −0.0069 → −0.0014 | interim paper (`main_submitted_final_backup.tex`) vs `A4/e1_tests_minilm.csv` |
| P2 −0.0044 (5), −0.0049 (6); BGE-s/BGE-b +0.003/+0.004 (5), +0.001 (6) | `P2X` (per-dataset deltas match `A4/e2_tests.csv` exactly) |
| P4 BGE-b / Tier B router − fixed +0.0016…+0.0028, p 0.03–0.16 | `EX2` P4_A_bgebase_std_k8_router_fixed, P4_B_bge_std_k8_router_fixed, P4_B_bgebase_std_k8_router_fixed |
| split-half BGE-b +0.015 vs +0.018 | `EX2` splithalf.bgebase all8 / insample |
| headroom concentration BGE-b 2–25%, 18–88% | `EX2` headroom_conc.bgebase |
| few-shot BGE-b +0.003/+0.009/+0.012/+0.014; excl Touché −0.003/+0.002/+0.006/+0.007; ArguAna share | `EX2` fewshot bgebase/k*; `A4/e4_fewshot_dataset_selection.csv` (bgebase k=20 arguana 0.049) |
| Table VI new rows | `EX2` (router, splithalf, fewshot) |
| Touché judged@10 BM25 0.47, dense 0.25–0.31; condensed 0.824 vs 0.643; TREC-COVID 0.91 both; NFCorpus judged@100 0.078–0.087; Quora condensed ≈1 | `A3/w9_judged.csv` |

## 2026-11-03 few-label alpha tuning (exploratory; rule in `phase2/05_alpha_fewshot/DECLARATION.md`, sha256 + time in `DECLARATION.sha256`, written before the run)

| Number | Source (`phase2/05_alpha_fewshot/results.json`) |
|---|---|
| BGE-b k=20 +0.021 [+0.011, +0.028]; −Touché +0.013 | bgebase/k20 macro, lo, hi, excl_touche |
| −Touché −ArguAna +0.006 | mean of bgebase/k20 per_ds over the 6 other datasets (0.0379/6) |
| k=10 +0.017; BGE-s +0.019; MiniLM +0.013 | bgebase/k10, bge/k20, minilm/k20 macro |
| three-system choice +0.012 | system_choice_ref bgebase/k20 |
| hindsight optimum +0.027 | `A4/e1_perdataset_bgebase.csv` MACRO oracle_ds − H (0.0274) |
| alpha 0.1–0.4, Touché 0.9 | bgebase/k20 median_alpha (NFCorpus 0.3, TREC-COVID 0.4 also) |
| SciFact −0.003, Quora −0.002 | bgebase/k20 per_ds |

## 2026-11-03/04 E5 text router (pre-registered; Drive file PREREG_E5_text_router_2026-11-03.md, createdTime 2026-11-03T18:13:32Z)

| Number | Source (`phase2/06_text_router/results.json`, produced by `eval.py` from `preds/`) |
|---|---|
| 0.526 vs 0.548, Δ −0.022 [−0.031, −0.014], p 1.00 | bgebase: macro_router, macro_H, all.delta/lo/hi/p_one |
| −0.008 without Touché; −0.004 without Touché and TREC-COVID | bgebase excl_touche, excl_touche_covid |
| τ_B 0.03; τ_D ≈ 0 (−0.03) | bgebase tau_B_mean, tau_D_mean |
| Touché 88% dense, −0.126; ArguAna +0.048 | bgebase per_ds |
| Quora cap 1,500; settings | `phase2/06_text_router/PREREGISTRATION.md`, `train.py` |
| secondaries BGE-s −0.017 [−0.024, −0.009], MiniLM −0.028 [−0.035, −0.020] | results.json bge / minilm all |

## 2026-11-05 E7 disagreement-guided labelling (pre-registered: Drive PREREG_E7_disagreement_labelling_2026-11-05.md, createdTime 2026-11-05T09:15:51Z; deviation in `phase2/07_disagreement/DEVIATIONS.md`)
| Number | Source (`phase2/07_disagreement/results.json`) |
|---|---|
| +0.002 over random at k=20 (BGE-b); 34% of random draws ≥ | bgebase/k20 diff 0.0019, share_rand_ge_DIS 0.340 |
| losses with BGE-s and MiniLM | bge/k20 diff −0.0079; minilm/k20 diff −0.0064 |
