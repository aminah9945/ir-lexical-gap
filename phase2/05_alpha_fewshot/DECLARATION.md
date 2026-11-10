# Few-label per-collection alpha tuning: analysis rule, written BEFORE running (2026-11-03)

Status: EXPLORATORY. Written down locally before the run; not externally timestamped. Run once; results reported as obtained.

Data: `04_improvements/results/alpha_grid_{model}_{dataset}.csv.gz` (per-query nDCG@10 for
H_alpha = alpha*minmax(BM25) + (1-alpha)*minmax(D), alpha in {0, 0.1, ..., 1}). Models: minilm, bge, bgebase. 8 datasets.

Procedure, per model, per dataset, per k in {10, 20, 30}, 1,000 random draws (seed 42):
1. Draw k labelled queries uniformly without replacement.
2. Pick alpha_hat = argmax of mean nDCG@10 on those k queries. Ties go to the alpha closest to 0.5 (the default).
3. On the remaining queries, gain = mean nDCG(H_alpha_hat) - mean nDCG(H_0.5).
4. Per-dataset gain = mean over draws. Headline = macro mean over 8 datasets.
   Also reported: the same without Touche, and without Touche and TREC-COVID;
   the share of draws with negative macro gain;
   and the comparison with the existing few-shot B/D/H system choice (e4_fewshot_dataset_selection.csv).

Primary contrast (stated in advance): BGE-b, k = 20, macro gain over alpha = 0.5.
"Useful" means a macro gain >= +0.005 that is still >= +0.005 without Touche.
Uncertainty: the 2.5-97.5% range of the macro gain across draws (sampling variability of the k labels).
