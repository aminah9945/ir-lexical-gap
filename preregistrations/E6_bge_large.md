# Pre-registration E6: a larger dense model (bge-large-en-v1.5, 335M parameters), written 2026-11-04 before encoding

## Encoding
- `laptop_kit/src/dense_runs.py bgelarge`: fp16, batch 32, max length 256.
- Same query prefix and top-K as bge-base (top-1000; Quora top-100).
- Sanity gate: the kit reproduced MiniLM SciFact nDCG@10 = 0.6451, identical to the paper.

## Analyses
No new tuning. All pipelines are rerun exactly as for bge-base, with D = bge-large:
1. QPP router (Tier A, std, selected variant, LODO).
2. Effect sizes.
3. Few-label alpha tuning (`05_alpha_fewshot` rule).
4. Text router (E5 settings, unchanged).

## Hypotheses and decision rules (same as before)
- **H-route.** The macro gain of the router over always-hybrid is at least +0.005 AND the one-sided sign-flip p < 0.05.
  - Prediction: not supported, as with the other three dense models.
- **H-alpha (exploratory).** The k = 20 alpha-tuning gain is at least +0.005 and is still at least +0.005 without Touche.
- **H-hybrid.** H(bge-large) - bge-large, pooled DerSimonian-Laird.
  - Reported whatever the outcome.
  - Prediction: the hybrid's gain over the dense model alone shrinks further as the model gets larger.

## Other commitments
- All results are reported as obtained.
- If the encoding does not finish before the paper deadline, E6 is left for the next version. No partial-dataset results will be reported as if complete.
