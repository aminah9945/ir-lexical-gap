# Reading list (verified) -- Phase 2, Agent 2

Every item below was checked against a primary record on 2026-10-21: the Crossref DOI record (the DOI registry), an arXiv abs page, the ACL Anthology, a Springer chapter page, a NeurIPS proceedings page, the TREC/NIST site, or a Hugging Face model card. dl.acm.org is blocked from this environment, so ACM papers were checked through their Crossref DOI record only.

- **Bib key** refers to `references.bib`.
- **PDF** is a direct open PDF that was confirmed to exist (HTTP 200, application/pdf, or an arXiv PDF). Agent 5 can download it.
- **DOI only** means no verified open PDF. The paper sits behind ACM, Springer or Elsevier, so the student fetches it through the university library.
- **Priority:** A = must read and cite, B = cite, C = background/tangential.

## 1. Classical IR / BM25
| Pri | Key | Paper | PDF |
|---|---|---|---|
| A | robertson2009prf | Robertson & Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond, FnTIR 3(4), 2009 | https://staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf (author-hosted) |
| B | robertson1976relevance | Robertson & Sparck Jones, Relevance weighting of search terms, JASIS 27(3), 1976 | DOI only: https://doi.org/10.1002/asi.4630270302 |
| B | salton1988term | Salton & Buckley, Term-weighting approaches in automatic text retrieval, IP&M 24(5), 1988 | DOI only: https://doi.org/10.1016/0306-4573(88)90021-0 |
| B | lavrenko2001relevance | Lavrenko & Croft, Relevance based language models, SIGIR 2001 (RM1/RM3 basis) | DOI only: https://doi.org/10.1145/383952.383972 |
| B | abduljaleel2004umass | Abdul-Jaleel et al., UMass at TREC 2004: Novelty and HARD (source usually cited for RM3) | https://trec.nist.gov/pubs/trec13/papers/umass.novelty.hard.pdf |

## 2. BEIR, reproducibility, dense-vs-lexical analyses
| Pri | Key | Paper | PDF |
|---|---|---|---|
| A | thakur2021beir | Thakur et al., BEIR, NeurIPS 2021 Datasets & Benchmarks | https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf ; https://arxiv.org/pdf/2104.08663 |
| A | kamalloo2024resources | Kamalloo et al., Resources for Brewing BEIR: Reproducible Reference Models and Statistical Analyses, SIGIR 2024 | https://arxiv.org/pdf/2306.07471 (earlier version, titled "...and an Official Leaderboard"; the SIGIR version is DOI only) |
| A | thakur2024touche | Thakur et al., Systematic Evaluation of Neural Retrieval Models on the Touché 2020 Argument Retrieval Subset of BEIR, SIGIR 2024 | https://arxiv.org/pdf/2407.07790 |
| A | formal2022match | Formal, Piwowarski, Clinchant, Match Your Words! A Study of Lexical Matching in Neural IR, ECIR 2022 (short) | https://arxiv.org/pdf/2112.05662 |
| B | sciavolino2021simple | Sciavolino et al., Simple Entity-Centric Questions Challenge Dense Retrievers, EMNLP 2021 | https://aclanthology.org/2021.emnlp-main.496.pdf |
| B | ren2023thorough | Ren et al., A Thorough Examination on Zero-shot Dense Retrieval, Findings of EMNLP 2023 | https://aclanthology.org/2023.findings-emnlp.1057.pdf |
| B | lin2021pyserini | Lin et al., Pyserini, SIGIR 2021 | https://arxiv.org/pdf/2102.10073 |
| B | muennighoff2023mteb | Muennighoff et al., MTEB, EACL 2023 | https://aclanthology.org/2023.eacl-main.148.pdf |
| C | formal2021splade | Formal et al., SPLADE, SIGIR 2021 (short) | https://arxiv.org/pdf/2107.05720 |
| C | ma2021replication | Ma et al., A Replication Study of Dense Passage Retriever (arXiv 2021) | https://arxiv.org/pdf/2104.05740 |
| C | lin2021fewbrief | Lin & Ma, A Few Brief Notes on DeepImpact, COIL, ... (arXiv 2021) | https://arxiv.org/pdf/2106.14807 |

## 3. Vocabulary mismatch
| Pri | Key | Paper | PDF |
|---|---|---|---|
| A | furnas1987vocabulary | Furnas et al., The vocabulary problem in human-system communication, CACM 30(11), 1987 | DOI only: https://doi.org/10.1145/32206.32212 |
| A | zhao2012automatic | Zhao & Callan, Automatic term mismatch diagnosis for selective query expansion, SIGIR 2012 | DOI only: https://doi.org/10.1145/2348283.2348354 |
| B | zhao2010term | Zhao & Callan, Term necessity prediction, CIKM 2010 | DOI only: https://doi.org/10.1145/1871437.1871474 |

## 4. Query performance prediction (pre- and post-retrieval)
| Pri | Key | Paper | Predictor(s) | PDF |
|---|---|---|---|---|
| A | cronentownsend2002predicting | Cronen-Townsend, Zhou, Croft, Predicting query performance, SIGIR 2002 | Clarity | DOI only: https://doi.org/10.1145/564376.564429 |
| A | zhou2007query | Zhou & Croft, Query performance prediction in web search environments, SIGIR 2007 | WIG | DOI only: https://doi.org/10.1145/1277741.1277835 |
| A | shtok2012predicting | Shtok et al., Predicting query performance by query-drift estimation, TOIS 30(2), 2012 | NQC | DOI only: https://doi.org/10.1145/2180868.2180873 |
| A | tao2014query | Tao & Wu, QPP by considering score magnitude and variance together, CIKM 2014 (short) | SMV | DOI only: https://doi.org/10.1145/2661829.2661906 |
| A | meng2023qpp | Meng, Arabzadeh, Aliannejadi, de Rijke, QPP: From Ad-hoc to Conversational Search, SIGIR 2023 | Gives formulas and default k for WIG/NQC/SMV/Clarity with BM25 and dense scores (used for our spec) | https://arxiv.org/pdf/2305.10923 |
| A | faggioli2023qpp | Faggioli et al., QPP for Neural IR: Are We There Yet?, ECIR 2023 | QPP is significantly worse on neural IR | https://arxiv.org/pdf/2302.09947 |
| B | he2004inferring | He & Ounis, Inferring query performance using pre-retrieval predictors, SPIRE 2004 | SCS, AvIDF, AvICTF | DOI only: https://doi.org/10.1007/978-3-540-30213-1_5 |
| B | zhao2008effective | Zhao, Scholer, Tsegay, Effective pre-retrieval QPP using similarity and variability evidence, ECIR 2008 | SCQ, VAR | DOI only: https://doi.org/10.1007/978-3-540-78646-7_8 |
| B | hauff2008survey | Hauff, Hiemstra, de Jong, A survey of pre-retrieval QPP predictors, CIKM 2008 (short) | survey | DOI only: https://doi.org/10.1145/1458082.1458311 |
| B | carmel2010estimating | Carmel & Yom-Tov, Estimating the Query Difficulty for IR, Morgan & Claypool 2010 | book | DOI only: https://doi.org/10.2200/S00235ED1V01Y201004ICR015 |
| B | faggioli2021enhanced | Faggioli et al., An Enhanced Evaluation Framework for QPP (sMARE), ECIR 2021 | evaluation | DOI only: https://doi.org/10.1007/978-3-030-72113-8_8 |
| C | arabzadeh2021bertqpp | Arabzadeh, Khodabakhsh, Bagheri, BERT-QPP, CIKM 2021 (short) | supervised neural QPP | DOI only: https://doi.org/10.1145/3459637.3482063 |

## 5. Selective / adaptive retrieval and routing between sparse and dense
| Pri | Key | Paper | PDF |
|---|---|---|---|
| A | arabzadeh2021predicting | Arabzadeh, Yan, Clarke, Predicting efficiency/effectiveness trade-offs for dense vs. sparse retrieval strategy selection, CIKM 2021. **Closest prior work to W1:** a per-query classifier choosing sparse/dense/hybrid from the query alone, on MS MARCO. | https://arxiv.org/pdf/2109.10739 |
| A | khramtsova2023selecting | Khramtsova et al., Selecting which Dense Retriever to use for Zero-Shot Search, SIGIR-AP 2023 (collection-level selection; unsupervised methods ineffective) | https://arxiv.org/pdf/2309.09403 |
| B | amati2004query | Amati, Carpineto, Romano, Query difficulty, robustness, and selective application of query expansion, ECIR 2004 | DOI only: https://doi.org/10.1007/978-3-540-24752-4_10 |
| B | cronentownsend2004framework | Cronen-Townsend, Zhou, Croft, A framework for selective query expansion, CIKM 2004 (poster) | DOI only: https://doi.org/10.1145/1031171.1031220 |
| C | mallen2023when | Mallen et al., When Not to Trust Language Models (adaptive retrieval), ACL 2023 | https://aclanthology.org/2023.acl-long.546.pdf |
| C | jeong2024adaptive | Jeong et al., Adaptive-RAG, NAACL 2024 | https://aclanthology.org/2024.naacl-long.389.pdf |

## 6. Hybrid fusion
| Pri | Key | Paper | PDF |
|---|---|---|---|
| A | bruch2024fusion | Bruch, Gai, Ingber, An Analysis of Fusion Functions for Hybrid Retrieval, TOIS 42(1), 2024. Convex combination beats RRF; RRF is sensitive to k; CC is sample-efficient. | https://arxiv.org/pdf/2210.11934 |
| A | cormack2009rrf | Cormack, Clarke, Büttcher, Reciprocal rank fusion..., SIGIR 2009 (short) | DOI only: https://doi.org/10.1145/1571941.1572114 |
| A | wang2021bert | Wang, Zhuang, Zuccon, BERT-based Dense Retrievers Require Interpolation with BM25, ICTIR 2021 (no arXiv version found) | DOI only: https://doi.org/10.1145/3471158.3472233 |
| B | chen2022out | Chen et al., Out-of-Domain Semantics to the Rescue! Zero-Shot Hybrid Retrieval Models, ECIR 2022 | https://arxiv.org/pdf/2201.10582 |
| C | kuzi2020leveraging | Kuzi et al., Leveraging Semantic and Lexical Matching ... A Hybrid Approach (arXiv 2020) | https://arxiv.org/pdf/2010.01195 |

## 7. Domain-specific retrieval, 2025 (proposal's evidence base; see 02_verification.md for accuracy issues)
| Pri | Key | Paper | What it actually is | PDF |
|---|---|---|---|---|
| A | mori2025assessing | Mori, de Oliveira, Yih, Ventresca, ICAIL 2025, pp. 114-128 | Full paper: BM25 vs dense on CJEU passage retrieval | https://arxiv.org/pdf/2506.12895 |
| C | cunha2025medlink | Cunha et al., MedLink, ECIR 2025, pp. 72-77 | **Demo paper**: bi-encoder+cross-encoder, 10 reports judged by one physician | Springer PDF (may need access): https://link.springer.com/content/pdf/10.1007/978-3-031-88720-8_13.pdf |
| C | ateia2025bioragent | Ateia & Kruschwitz, BioRAGent, ECIR 2025, pp. 1-5 | **Demo paper**: LLM query expansion RAG system; no evaluation in abstract | Springer PDF (may need access): https://link.springer.com/content/pdf/10.1007/978-3-031-88720-8_1.pdf |
| C | doostmohammadi2025studying | Doostmohammadi & Kuhlmann, EMNLP 2025, pp. 26835-26844 | RALM **training** efficiency vs query-neighbour overlap (perplexity) | https://aclanthology.org/2025.emnlp-main.1363.pdf |
| C | akarsu2026bm25 | Akarsu, Karaman, Mierbach, From BM25 to Corrective RAG (arXiv 2026, preprint) | Financial text+table QA; BM25 > dense | https://arxiv.org/pdf/2604.01733 |
| C | pandey2026domain | Pandey, Domain-Adaptive and Scalable Dense Retrieval for Content-Based Recommendation (arXiv 2026, preprint) | | https://arxiv.org/pdf/2602.00899 |
| C | sager2025deep | Sager, Kamaraj, Grewe, Stadelmann, Deep Retrieval at CheckThat! 2025 (CLEF WN, arXiv) | | https://arxiv.org/pdf/2505.23250 |

## 8. Dense retrievers used (MiniLM / sentence-transformers, BGE, E5)
| Pri | Key | Paper / model card | PDF |
|---|---|---|---|
| A | reimers2019sbert | Reimers & Gurevych, Sentence-BERT, EMNLP-IJCNLP 2019 | https://aclanthology.org/D19-1410.pdf |
| A | wang2020minilm | Wang et al., MiniLM, NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf |
| A | xiao2024cpack | Xiao et al., C-Pack (BGE models), SIGIR 2024 | https://arxiv.org/pdf/2309.07597 |
| B | wang2022e5 | Wang et al., E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training (arXiv 2022) | https://arxiv.org/pdf/2212.03533 |
| A | hf_allminilm / hf_bgesmall / hf_bgebase / hf_e5base | Model cards (exact checkpoints) | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 ; https://huggingface.co/BAAI/bge-small-en-v1.5 ; https://huggingface.co/BAAI/bge-base-en-v1.5 ; https://huggingface.co/intfloat/e5-base-v2 |

## 9. Statistics and agreement
| Pri | Key | Paper | PDF |
|---|---|---|---|
| A | smucker2007comparison | Smucker, Allan, Carterette, A comparison of statistical significance tests for IR evaluation, CIKM 2007 | DOI only: https://doi.org/10.1145/1321440.1321528 |
| B | webber2010rbo | Webber, Moffat, Zobel, A similarity measure for indefinite rankings (RBO), TOIS 28(4), 2010 | DOI only: https://doi.org/10.1145/1852102.1852106 |
| B | dersimonian1986meta | DerSimonian & Laird, Meta-analysis in clinical trials, CCT 7(3), 1986 | DOI only: https://doi.org/10.1016/0197-2456(86)90046-2 |
| B | cohen1960coefficient | Cohen, A coefficient of agreement for nominal scales, EPM 20(1), 1960 | DOI only: https://doi.org/10.1177/001316446002000104 |

## Not verified, so kept out of the bib
- **Holm (1979)**, "A simple sequentially rejective multiple test procedure", Scand. J. Statist. 6(2):65-70. Confirmed only through secondary sources. JSTOR (stable/4615733) returned a bot challenge, and 10.2307/4615733 is not a registered DOI. It is canonical and safe to cite, but someone must open the JSTOR page manually before it goes into the bib.
- **He & Ounis**, "A Query-based Pre-retrieval Model Selection Approach to Information Retrieval", RIAO 2004, pp. 706-719. Only secondary listings were found (IR Anthology, a citing arXiv paper).
- **Le Zhao's CMU PhD thesis** on term mismatch was not located.
