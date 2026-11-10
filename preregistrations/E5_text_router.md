# Pre-registration: text-based query router (E5), written 2026-11-03 before any run

Uploaded to the student's Google Drive (server-side createdTime = external timestamp) BEFORE data download, training or evaluation.

## Question
Can a router that reads the query TEXT, rather than QPP features, beat the fixed hybrid (always-H, alpha = 0.5), zero-shot across collections?

## Data and labels
- 8 BEIR test collections. Per-query nDCG@10 for B (BM25), D and H are taken from
  `03_experiments/results/w1_router_perquery/A_{model}_std_k8.csv.gz` (columns B, D, H).
- Targets: Delta_B = B - H and Delta_D = D - H. Query text comes from BEIR `queries.jsonl`.

## Model
- `sentence-transformers/all-MiniLM-L6-v2` encoder (mean pooling) with a linear head with 2 outputs (Delta_B, Delta_D).
- Fine-tuned end to end with MSE loss. AdamW, lr 3e-5, 2 epochs, batch 32, max length 128 tokens, seed 42. CPU.
- Training is leave-one-dataset-out (LODO): train on 7 collections, test on the 8th.
- Quora is subsampled to 1,500 training queries (seed 42), so it does not dominate. It is always evaluated on all 10,000.
- No hyperparameter search. These settings are fixed now.

## Decision rule (identical to the pre-declared QPP "selected" router)
Choose argmax{0, predicted Delta_B, predicted Delta_D}.
Within 0.005 of the maximum, prefer the cheapest system (B < D < H).

## Primary test
- D = BGE-b (bge-base-en-v1.5). Macro nDCG@10 of the router minus always-H over 8 collections.
- Success requires BOTH:
  - the macro gain is at least +0.005, AND
  - the one-sided stratified sign-flip test (10^4 flips) gives p < 0.05.
- Secondary, uncorrected and labelled as such: D = BGE-s and D = MiniLM.

## Also reported, whatever the outcome
- Stratified bootstrap 95% CI.
- Results without Touche, and without Touche and TREC-COVID (post hoc).
- Within-collection Kendall tau between predicted and true Delta_B.
- The routing mix.
- Comparison with the QPP router.
