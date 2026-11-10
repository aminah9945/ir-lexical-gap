# REVISION_LOG.md: response to `phase2/08_critique.md` (Agent 7, 2026-10-22)

## Overview

**New numbers.** All new numbers come from `analysis_extra/extra_numbers.py` and are stored in `analysis_extra/extra_numbers.json`.
- They are computed only from existing per-query result files; nothing was tuned.
- Everything added after the review is labelled post hoc or exploratory in the paper. The one exception is the micro average, which was pre-declared as a secondary aggregate.
- Every new number is traced in `numbers_trace.md`, in the section "Added in revision".

**Bootstrap settings.**
- Stratified percentile bootstrap: queries are resampled within each collection, 10^4 resamples, seed 42.
- The p-values use the project's `perm_test_macro`.
- The critic used 5,000 resamples, so a few CI bounds differ in the third decimal. Example: the best-variant CI is [−0.002, +0.005], as the critic found. The four Table IV CIs match the critic's.

**Page count.** The paper went from 9 to **7 pages**:
- the body ends on page 6;
- the references take the rest of page 6 and page 7.

Reaching 6 pages including references would require dropping substance, specifically the sensitivity table, a figure, or Related Work. We therefore took the orchestrator's fallback: 6 pages plus 1 page of references. Trims made:
- removed two figures (router bars and win-share);
- cut Related Work by about 45%;
- compressed the feature list and the setup;
- removed the RAG, demo-paper and off-topic citations;
- cut the references from 49 to 30, all from the verified bib.

## MUST items

| Item | Status | What changed |
|---|---|---|
| MUST-1 "more than any query-level router" | Done | Abstract now says "more than any label-free router" and adds that almost all of the gain comes from Touché-2020, whose pool favours BM25. |
| MUST-2 Touché sensitivity of the router | Done | Table IV gains a "95% CI" column; all 4 CIs match the critic's values. The text gives +0.002 [−0.003, +0.006] without Touché and +0.004 [−0.001, +0.008] (p = 0.06) without Touché and TREC-COVID, and states that a gain up to +0.008 is not excluded. The abstract carries the six-collection result. New Table VI (`tab:sens`) collects the sensitivities. |
| MUST-3 Touché dependence of the few-shot result | Done | Sec. IV-F gives the without-Touché values (BGE-s −0.006/0.000/+0.004/+0.006; MiniLM negative for k ≤ 30) and the judgement-bias caveat. The guideline is rewritten ("detect the rare collection where one system dominates … otherwise expect no gain"), and so are the conclusion and the Fig. 2 caption. |
| MUST-4 micro average | Done | One sentence: 0.765 vs 0.761 (+0.004), carried by ArguAna, "not a win under the macro headline". |
| MUST-5 macro vs pooled in Table III | Done | Added a "Macro" column. The caption defines Macro and Pooled, and Sec. III defines "Pooled". The text reads "+0.005 pooled (−0.002 macro)". |
| MUST-6 noise calibration of the oracle | Done, with a caveat | New subsection "How much of the oracle headroom is real?". It reports the BM25 vs no-stemming oracle (+0.043; macro difference 0.0004) and BM25 vs WordNet (+0.013). Caveat added beyond the critique: stemming differs by up to 0.09 on single collections (Touché), so this is a reference point, not a pure noise floor; the text says so. The abstract and conclusion now say that the headroom is "inflated by per-query noise". |
| MUST-7 bounds, not only non-significance | Done | Best variant: 95% CI [−0.002, +0.005], "at most the pre-declared minimum gain". Table V gains a CI column for P1–P4. P4 router − fixed CI [−0.003, +0.002] (also covers COULD-7). |
| MUST-8 `selected_argmax` is post hoc | Done | Labelled "(added post hoc)" in Sec. III and listed under Post-hoc analyses in Threats. |
| MUST-9 logistic collapse | Done | Now reads "mostly collapses to always-dense (≥91% dense in five of seven configurations)". |
| MUST-10 Faggioli non-replication | Done | Same-five-collection comparison: s1−s10 0.30 vs 0.24, WIG 0.14 vs 0.17, NQC 0.16 vs 0.19, "partly in line with" Faggioli. `04_interpretation.md` and `06` were **not** edited (outside my remit); flagged for the orchestrator. |
| MUST-11 Faggioli qualifier | Done | Related Work adds "in lexical-oriented scenarios"; the Discussion says "consistent with". |
| MUST-12 H1 provenance | Done | "fixed, with decision rules, in the Phase-1 analysis code; H2 is the motivating question of the project proposal". "Before evaluation" is not claimed, since there is no git history to show it. |
| MUST-13 verifiable pre-declarations | Done (text) | The paper says the logs "are local files with no external timestamp; they are not registered with a third party". The project directory is not a git repository, so no commit timestamps exist. The student should commit the logs to the public release; the paper does not claim timestamps. |
| MUST-14 AI-use disclosure | Done | Rewritten per the orchestrator's account. The AI assistant (Claude, via Claude Code) designed and ran the experiments and analyses, wrote the code and the pre-declarations, verified the literature, and drafted and revised the manuscript, under the author's direction. The author directed the work, ran the Colab notebooks and takes full responsibility for the content. It is merged into "Acknowledgment and Use of AI Tools", as IEEE places AI disclosure in the acknowledgements. |
| MUST-15 author block and acknowledgement placeholders | **Not resolved (needs the student)** | Kept as placeholders, as instructed. The student must fill them in (single-blind IEEE regional venue). |
| MUST-16 release placeholder | **Not resolved (needs the student)** | There is still no repository or DOI. The student must create the public repository (or a Zenodo DOI) and insert it. |
| MUST-17 page budget | Done (6 + 1 refs) | See the overview. Two figures were deleted and Related Work was cut. References went from 49 to 30. The feature list moved to "the code release". The title is shortened to three lines; it still has three lines in the preview. Table II stays single-column for now; PENDING item 5 notes it may need `table*` once BGE-b columns are added. |

## SHOULD items

| Item | Status | What changed |
|---|---|---|
| SHOULD-1 title | Done | "Query-Performance Signals Do Not Route Between Sparse, Dense and Hybrid Retrieval: A Pre-Declared Negative Result on BEIR". |
| SHOULD-2 overclaiming words | Done | "rigorously" → "under pre-declared decision rules". "The entire gain" → "Essentially all of the gain". "Collection-level choice is easy" is gone. "Realises any of it" → "realises its query-level part". Added "plausibly because". "Quantify pooling bias" removed. |
| SHOULD-3 small-collection weighting and sensitivities | Done | Sec. III explains the macro weighting. Table VI covers the router (both D), P4, CE(BM25), split-half and few-shot. P1–P3 use five collections with no small ones; this is stated. |
| SHOULD-4 pool bias vs the prior-shift explanation | Done (main part) | The prior-shift bullet says the prior "may itself be partly a pooling artefact: dense judged@k on Touché is pending". The BM25-LODO judged@10 detail was omitted for space. |
| SHOULD-5 in-domain ceiling on small collections | Done | Threats: "the in-domain tree router is constant by construction (40-query folds, leaf size 50)". |
| SHOULD-6 scope of the "Nor do" list | Done | The abstract now says "On five collections …" and "(on ≤150 queries per collection)". |
| SHOULD-7 "seven configurations" | Done | Now reads "(four feature/model settings, three with percentile features)". |
| SHOULD-8 recall@100 wording | Done | Now reads "higher recall@100 on all eight (7 Holm-significant)". |
| SHOULD-9 Kamalloo exceptions incl. SciDocs | Done | Now reads "Touché, ArguAna, Quora and SciDocs … largely coincide". |
| SHOULD-10 Kuzi scope | Done | The citation was dropped entirely, as part of the reference trim. |
| SHOULD-11 CE sign agreement | Done | Now reads "matches BEIR's direction on seven collections and is null on ArguAna in both", plus the macro bootstrap CI [+0.009, +0.041]. The project's bootstrap CI is used instead of the DL CI the critic suggested; the DL CI is anti-conservative. |
| SHOULD-12 DL with k = 5–8 | Done (statement) | The paper says the DL intervals are anti-conservative and that the macro bootstrap is primary. HKSJ was not computed (COULD-level effort; not needed for the claims). |
| SHOULD-13 cite Holm (1979) | **Not done** | It is not in the verified bib, and the instructions forbid unverified references. JSTOR and dl.acm are blocked from this environment. The bootstrap type is now named (percentile). The student can verify and add Holm from the university network. |
| SHOULD-14 related work on learned/LLM routing | Partly done | Added an uncited sentence: "Our result concerns cheap QPP features; routers built on learned query encoders or large language models are not tested here." The suggested papers (Balasubramanian & Allan 2010; RouterRetriever; DAT; He & Ounis 2004) are not in the verified bib, so they are not cited. Agent 2 or the student should verify them first. |

## COULD items

| Item | Status |
|---|---|
| COULD-1 permutation-floor p | Done: P4 is written as "< 10^-3" |
| COULD-2 x-tick labels | Not done (Fig. 1 uses rotated labels; still readable) |
| COULD-3 Fig. 2 legend / without-Touché curve | Not done; the without-Touché values are in Table VI and the text |
| COULD-4 ArguAna 1,401 vs 1,406 footnote | Not done (space) |
| COULD-5 "within 0.005 at three decimals" | Done |
| COULD-6 per-collection τ | Not done |
| COULD-7 P4 router − fixed CI | Done ([−0.003, +0.002]) |
| COULD-8 trace note on 3.4× | Done (`numbers_trace.md`) |
| COULD-9 Colab provenance of BGE-s rankings | Done (Sec. III, Systems) |

## Other changes

- Removed `\IEEEoverridecommandlockouts` and `multirow`, which were unused.
- All original `% PENDING-COLAB` markers are kept, and new ones were added for the abstract and `tab:sens` (16 in total). No pending value was invented.
