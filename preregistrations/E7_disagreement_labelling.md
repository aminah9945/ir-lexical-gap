# Pre-registration E7: disagreement-guided labelling for per-collection alpha tuning (written 2026-11-05, before any run)

## Question
At the same labelling budget k, does labelling the queries on which BM25 and the dense model disagree most pick a better per-collection hybrid weight alpha than labelling random queries?

## Data
- Per-query nDCG@10 for H_alpha, alpha in {0, 0.1, ..., 1}: `04_improvements/results/alpha_grid_{model}_{ds}.csv.gz`.
- Disagreement score: `rbo100` (rank-biased overlap between the BM25 and dense top-100 lists), from `03_experiments/results/qpp/tierB_{model}_{ds}.csv`. It uses no relevance labels.
- 8 BEIR collections.

## Method (DIS)
1. In each collection, sort the queries by rbo100, ascending (most disagreement first). Ties go by qid as a string.
2. Label the first k queries.
3. Choose alpha_hat = argmax of mean nDCG@10 on those k queries. Ties go to the alpha closest to 0.5.
4. Gain = mean nDCG(H_alpha_hat) - mean nDCG(H_0.5), over the remaining (unlabelled) queries.
5. Headline = macro mean over the 8 collections.

## Baseline (RAND)
The identical procedure with k random queries: 1,000 draws, seed 42, gain averaged over draws. This is the same as `05_alpha_fewshot`.

## Primary test
- Model: BGE-b (bgebase). k = 20.
- Success requires BOTH:
  - macro(DIS) - macro(RAND) >= +0.003, AND
  - the same difference without Touche is >= +0.003.
- Also reported: the share of the 1,000 random draws whose macro gain is >= DIS's macro gain (a percentile, showing where DIS falls in the random distribution).

## Secondary (uncorrected, labelled as such)
k = 10; BGE-s; MiniLM.

## Commitment
One method, run once, reported whatever the outcome. No other selection rules will be tried for the paper.
