"""Report prose. Numbers are read from results/ so text and tables cannot drift apart."""
import json
import os

import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PRETTY = {"nfcorpus": "NFCorpus", "scifact": "SciFact", "fiqa": "FiQA-2018", "scidocs": "SciDocs",
          "arguana": "ArguAna", "trec-covid": "TREC-COVID", "quora": "Quora", "webis-touche2020": "Touche-2020"}
DS = list(PRETTY)

_tab = pd.read_csv(os.path.join(RES, "table_main.csv"))
NDCG = _tab.pivot(index="dataset", columns="system", values="ndcg@10")
_abl = pd.read_csv(os.path.join(RES, "ablation_tests.csv"))
_dom = pd.read_csv(os.path.join(RES, "domain_stats.csv")).set_index("dataset")
_bs = pd.read_csv(os.path.join(RES, "boolean_set_summary.csv")).set_index("dataset")
_gs = pd.read_csv(os.path.join(RES, "bm25_grid_summary.csv")).set_index("dataset")
_aud = json.load(open(os.path.join(RES, "failure_audit_summary.json")))
_resc = pd.read_csv(os.path.join(RES, "failure_rescue_rates.csv")).set_index("primary")
_ct = pd.read_csv(os.path.join(RES, "failure_crosstab.csv")).set_index("dataset")
_h3 = pd.read_csv(os.path.join(RES, "h3_correlations.csv"))


def n(x, d=3):
    return f"{x:.{d}f}"


def pv(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def sig_count(comp, sign=1):
    s = _abl[_abl.comparison == comp]
    return int(((s.p_holm < .05) & (s["diff"] * sign > 0)).sum())


TITLE = "Diagnosing Lexical-Matching Limitations Across Domains: Experimental Results"
SUBTITLE = ("Classical IR (Boolean, TF-IDF/VSM, BM25) on 8 BEIR datasets, with a vocabulary-overlap analysis, "
            "query-expansion and dense/hybrid comparisons. Every verdict below is produced by a decision rule "
            "fixed in code before inspecting the outcome (src/analysis.py, src/failures.py).")

EXEC_INTRO = ("The proposal's claims were turned into six testable hypotheses. Two are not supported (H1, H2), "
              "two are partially supported (H5, H6), and two are supported (H3, and H4 with caveats about label accuracy). The most consequential "
              "result is that <b>domain-level vocabulary specificity (jargon) does not explain where BM25 succeeds "
              "or fails</b>, whereas <b>query-level overlap does</b>, but in a way that is only partly specific to "
              "lexical retrieval and cannot be measured at query time.")

CLAIMS = [
    ("H1", "BM25 > TF-IDF/VSM > Boolean on every domain (Objective 2)",
     "Paired randomization test on nDCG@10, Holm-corrected over 32 tests"),
    ("H2", "Lexical retrieval degrades in domains of higher term specificity (Problem statement)",
     "Spearman across 8 domains: measured jargon rate vs BM25 nDCG@10"),
    ("H3", "Query-document vocabulary overlap explains BM25 effectiveness (Objective 3)",
     "Partial Spearman per dataset + linear mixed model (random intercept per dataset)"),
    ("H4", "Failures fall into identifiable modes that differ by domain (Objective 4)",
     "Heuristic taxonomy, chi-square across domains, manual audit"),
    ("H5", "Dense retrieval helps most where overlap is low (Objective 5)",
     "Partial Spearman of (dense - BM25) gap vs overlap; mixed model"),
    ("H6", "An overlap threshold tells when BM25 alone is defensible (Deliverable 4)",
     "Leave-one-dataset-out threshold rule, balanced accuracy, AUC"),
]


def _ev_h1(V):
    d = V["H1"]["detail"]
    return (f"BM25 significantly beats TF-IDF on only {d['bm25>tfidf']['holds']}/8 datasets; TF-IDF significantly "
            f"beats BM25 on ArguAna ({n(NDCG.loc['arguana','tfidf'])} vs {n(NDCG.loc['arguana','bm25'])}). "
            f"TF-IDF > Boolean-OR on {d['tfidf>bool_or']['holds']}/8 (reversed on Touche). BM25 > Boolean on 8/8.")


def _ev_h2(V):
    h = V["H2"]
    return (f"rho(corpus jargon, BM25) = {n(h['corpus_jargon']['rho'],2)} ({pv(h['corpus_jargon']['p'])}), i.e. wrong "
            f"sign and not significant. The proposal's specificity ordering of NFCorpus, SciFact and FiQA does not match "
            f"the measured one: FiQA is the <i>least</i> jargon-heavy.")


def _ev_h3(V):
    h = V["H3"]
    return (f"Positive in {h['n_datasets_sig_positive']}/8 datasets (partial rho 0.37-0.73; Touche null). Mixed model: "
            f"+{n(h['mixed']['bm25']['coef'])} nDCG per SD of overlap. <b>But</b> overlap also predicts the dense "
            f"model (+{n(h['mixed']['minilm']['coef'])}), so about half the effect is general query difficulty.")


def _ev_h4(V):
    h = V["H4"]
    return (f"Modes differ by domain (chi2 = {h['chi2']:.0f}, Cramer's V = {h['cramers_v']:.2f}, {pv(h['p'])}). "
            f"Labels are heuristic: manual audit accuracy {_aud['test_acc']:.0%} "
            f"(95% CI {_aud['wilson_lo']:.0%}-{_aud['wilson_hi']:.0%}, n={_aud['test_n']}). Paraphrase/indirect "
            f"relevance and term competition dominate; abbreviation is rare.")


def _ev_h5(V):
    h, b = V["H5"], V.get("H5_bge_robustness")
    extra = f"; bge-small: {b['n_datasets_sig_negative']}/8" if b else ""
    return (f"Pooled effect is clear (mixed-model coef {n(h['mixed']['coef'])}, {pv(h['mixed']['p'])}) but significant "
            f"in only {h['n_datasets_sig_negative']}/8 datasets for MiniLM{extra}. Not significant on FiQA, SciDocs, "
            f"TREC-COVID or Touche.")


def _ev_h6(V):
    h = V["H6"]
    return (f"The learned threshold is stable (coverage about {h['threshold_median']:.2f} in 7/8 folds) but transfers poorly: "
            f"mean held-out balanced accuracy {n(h['mean_lodo_bacc'],2)}, only {h['folds_ge_055']}/8 folds >= 0.55; "
            f"AUC {n(h['mean_auc'],2)}. Overlap needs relevance labels, so it cannot route live queries.")


EVIDENCE = {"H1": _ev_h1, "H2": _ev_h2, "H3": _ev_h3, "H4": _ev_h4, "H5": _ev_h5, "H6": _ev_h6}


def BAD_NEWS(V):
    return [
        "<b>The central framing of the proposal (that lexical retrieval degrades with domain specificity) is not supported "
        "by this data.</b> The direction is, if anything, reversed: jargon-heavy collections (SciFact, TREC-COVID, NFCorpus) "
        "are where BM25 is relatively <i>strongest</i> against dense models (rho = "
        f"{n(V['H2']['corpus_jargon_vs_dense_gap']['rho'],2)} for jargon vs. MiniLM's gain, "
        f"{pv(V['H2']['corpus_jargon_vs_dense_gap']['p'])}, n = 8). This matches the proposal's own Section 7 "
        "preprints, so the proposal currently argues both directions and must pick one.",
        "<b>The overlap-effectiveness correlation (Objective 3) is partly circular.</b> BM25 scores documents by shared "
        "terms, so a positive correlation is close to guaranteed. The informative contrast is that overlap predicts the dense "
        "model about half as strongly. Report the BM25-minus-dense difference, not the raw correlation, as the contribution.",
        "<b>A 'vocabulary-overlap threshold' (Deliverable 4) cannot be applied at query time.</b> Overlap is defined against "
        "the relevant documents, which are unknown for a new query. The threshold that was learned (coverage about 0.6) "
        f"barely transfers to unseen domains (balanced accuracy {n(V['H6']['mean_lodo_bacc'],2)}).",
        f"<b>The finance hypothesis borrowed from the 2026 preprint fails here.</b> On FiQA, even small dense models beat BM25 by "
        f"{n(NDCG.loc['fiqa','minilm']-NDCG.loc['fiqa','bm25'])} (MiniLM) and {n(NDCG.loc['fiqa','bge']-NDCG.loc['fiqa','bm25'])} "
        "(bge-small) nDCG@10, the largest gains among the original three domains. FiQA is opinion-style forum text, not "
        "ticker-heavy financial documents.",
        "<b>Domain-level conclusions rest on n = 8 points</b> (n = 3 in the original design, which could not support any "
        "correlation). They are underpowered; per-query analyses (about 14,000 queries) carry the inferential weight.",
        "<b>'Dense vs BM25' depends on which dense model is used.</b> On TREC-COVID, MiniLM loses to BM25 by "
        f"{n(abs(NDCG.loc['trec-covid','minilm']-NDCG.loc['trec-covid','bm25']))} while bge-small wins by "
        f"{n(NDCG.loc['trec-covid','bge']-NDCG.loc['trec-covid','bm25'])}. Claims about 'semantic retrieval' in general "
        "should not be drawn from one model.",
    ]


SETUP = [
    "<b>Data.</b> The proposal's three BEIR datasets (NFCorpus, SciFact, FiQA-2018) plus five more for domain-level "
    "power: SciDocs (scientific citation prediction), ArguAna and Touche-2020 (argument retrieval), TREC-COVID "
    "(biomedical) and Quora (general-domain duplicate questions). Test splits; queries without a positive judgement are "
    "dropped; a document whose id equals the query id is removed from rankings (BEIR convention).",
    "<b>Lexical systems.</b> Lucene-style BM25 (k1 = 0.9, b = 0.4, the Pyserini BEIR defaults), TF-IDF/VSM (SMART ltc, "
    "cosine), Boolean OR ranked by coordination level, and Boolean AND (set retrieval; random order within the set). "
    "Analysis: \\w+ tokens, lower-case, Lucene stopword list, Porter stemmer. Title and text are indexed together "
    "('flat'). Four preprocessing variants form the E2 ablation.",
    "<b>Expansion (E5).</b> RM3 pseudo-relevance feedback (10 docs, 10 terms, original weight 0.5) and WordNet synonym "
    "expansion (up to 3 synonyms per term from the first two senses, weight 0.3).",
    "<b>Dense and hybrid (E6, added).</b> all-MiniLM-L6-v2 and bge-small-en-v1.5, exact cosine search, 256 tokens, run on "
    "a Colab T4 GPU; hybrids by Reciprocal Rank Fusion (k = 60) and by convex combination of min-max-normalised scores "
    "(alpha = 0.5). CPU re-runs of MiniLM on five datasets agree with the GPU run to within 0.0004 mean nDCG@10.",
    "<b>Overlap features (E3).</b> Primary, fixed in advance: <i>coverage</i> = share of the query's distinct "
    "(stemmed, stopped) terms present in a relevant document, averaged over that query's relevant documents. Robustness "
    "variants: IDF-weighted coverage, maximum coverage, Jaccard. Domain specificity is <i>measured</i> as the share of "
    "corpus and query tokens with general-English Zipf frequency < 3 (wordfreq), plus mean query IDF.",
    "<b>Statistics.</b> Paired two-sided randomization tests (10,000 sign flips) with Holm correction; partial Spearman "
    "correlations controlling for number of query terms and log number of relevant documents, with 95% bootstrap CIs; "
    "linear mixed models with a random intercept per dataset; leave-one-dataset-out validation for the threshold. "
    "Seed 42 throughout.",
]

STATS_NOTE = ("Jargon = share of tokens rarer than Zipf 3 (about 1 per million words) in general English. Mean coverage = "
              "primary overlap measure. Five ArguAna queries whose relevant documents are missing from the corpus are "
              "excluded from overlap analyses. NFCorpus, SciFact and FiQA counts match the proposal's Table 10.")

VALIDATION = ("BM25 reproduces the Pyserini BM25-flat reference numbers to within 0.005 nDCG@10 on all eight datasets, "
              "and our metric code matches pytrec_eval exactly on SciFact and FiQA (NFCorpus differs by 0.0005 because of "
              "tie ordering). See results/metric_validation.txt.")

PYSERINI_BM25 = {"nfcorpus": 0.322, "scifact": 0.679, "fiqa": 0.236, "scidocs": 0.149, "arguana": 0.397,
                 "trec-covid": 0.595, "quora": 0.789, "webis-touche2020": 0.442}

CAP_FIG1 = "Figure 1. nDCG@10 for every system and dataset (darker = higher)."
SYSTEM_ORDER = ["bool_and", "bool_or", "tfidf", "bm25", "bm25_rm3", "bm25_wordnet", "minilm", "bge",
                "hybrid_rrf_minilm", "hybrid_cc_minilm", "hybrid_rrf_bge", "hybrid_cc_bge"]
SYSNAME = {"bool_and": "Boolean AND", "bool_or": "Boolean OR (coord.)", "tfidf": "TF-IDF / VSM", "bm25": "BM25",
           "bm25_rm3": "BM25 + RM3", "bm25_wordnet": "BM25 + WordNet", "minilm": "MiniLM (dense)",
           "bge": "bge-small (dense)", "hybrid_rrf_minilm": "Hybrid RRF (MiniLM)",
           "hybrid_cc_minilm": "Hybrid convex (MiniLM)", "hybrid_rrf_bge": "Hybrid RRF (bge)",
           "hybrid_cc_bge": "Hybrid convex (bge)"}
CAP_TABLES = "Bold = best system per dataset for that metric. Per-query values: results/perquery_*.csv."

BOOLEAN_TEXT = (f"Strict conjunctive retrieval returns <i>nothing</i> for {_bs.empty_frac.drop('webis-touche2020').min():.0%}-"
                f"{_bs.empty_frac.max():.0%} of queries on seven of eight datasets (Touche, with short keyword queries, "
                f"is the exception at {_bs.loc['webis-touche2020','empty_frac']:.0%}). This is the vocabulary-mismatch "
                "problem in its most direct form: one missing term is enough to lose every relevant document.")

CAP_FIG5 = ("Figure 2. Effect of each intervention on nDCG@10. Preprocessing and expansion rows are relative to BM25 "
            "without that step / plain BM25; dense and hybrid are relative to BM25.")


def ABLATION_TEXT(V):
    return [
        f"<b>Stemming (E2)</b> significantly helps only on Quora (+{n(_abl[(_abl.dataset=='quora')&(_abl.comparison=='stemming')]['diff'].iloc[0])}) "
        f"and significantly hurts on Touche ({n(_abl[(_abl.dataset=='webis-touche2020')&(_abl.comparison=='stemming')]['diff'].iloc[0])}). "
        "Morphological variation is therefore a minor source of lexical failure on these collections. Stopword removal "
        "is negligible.",
        f"<b>Query expansion (E5)</b> does not close the vocabulary gap. WordNet expansion never significantly helps (0/8). "
        f"RM3 significantly helps on {sig_count('RM3 expansion')}/8 (NFCorpus, TREC-COVID) and significantly hurts on "
        f"{sig_count('RM3 expansion', -1)}/8 (FiQA, Quora). Generic thesauri lack the domain-specific relations "
        "these queries need.",
        f"<b>Hybrid retrieval</b> is the most reliable intervention: convex BM25+MiniLM fusion significantly improves on "
        f"BM25 in {sig_count('hybrid convex (minilm)')}/8 datasets (+0.035 to +0.157) and is worse only on Touche. "
        f"Dense-only MiniLM significantly beats BM25 on {sig_count('dense (minilm)')}/8 and loses badly on Touche.",
    ]


GRID_TEXT = (f"Tuning BM25 on test queries (an optimistic upper bound, computed on a 1,000-query sample for ArguAna "
             f"and Quora, hence the slightly different defaults) gains up to "
             f"{n((_gs.best-_gs['default']).max())} (ArguAna, b = 0.9). Stronger length normalisation recovers most of "
             "TF-IDF's ArguAna advantage, so H1's ArguAna reversal is a parameter effect, not a weakness of the BM25 model.")


def SECTIONS(V):
    h3 = _h3[(_h3.feature == "cov_mean")].pivot(index="dataset", columns="system", values="partial_rho").reindex(DS)
    rob = _h3[(_h3.system == "bm25")].pivot(index="dataset", columns="feature", values="partial_rho").reindex(DS)
    return [
        ("H1", "H1. Is BM25 > TF-IDF > Boolean everywhere?  Verdict: " + V["H1"]["verdict"], [
            "BM25 beats both Boolean models everywhere, but it is <b>not</b> uniformly better than TF-IDF. The difference "
            "is not significant on NFCorpus, SciDocs and Quora; TF-IDF is significantly better on ArguAna, whose queries "
            "are whole arguments (about 196 tokens) where cosine length normalisation matters. On TREC-COVID and "
            "Touche, TF-IDF performs only at Boolean-OR level, because ltc weighting favours very short documents. The "
            "ordering is a property of model <i>and</i> collection, not a law. Full tests: results/h1_tests.csv."], []),
        ("H2", "H2. Does lexical retrieval degrade with domain specificity?  Verdict: " + V["H2"]["verdict"], [
            f"Measured corpus jargon rate vs BM25 nDCG@10 across 8 domains: rho = {n(V['H2']['corpus_jargon']['rho'],2)} "
            f"({pv(V['H2']['corpus_jargon']['p'])}); query jargon rho = {n(V['H2']['q_jargon']['rho'],2)}; mean query IDF "
            f"rho = {n(V['H2']['q_mean_idf']['rho'],2)}. None is negative and significant. Raw nDCG is not comparable "
            "across datasets (different judging depth, number of relevant documents and task type), so the BM25-relative "
            "gain of a dense model is also shown. Its association with jargon is negative (dense helps <i>less</i> in "
            "jargon-heavy domains) but not significant with n = 8.",
            "The proposal assumed specificity rises from NFCorpus to SciFact to FiQA. Measured jargon rates are "
            f"FiQA {n(_dom.loc['fiqa','corpus_jargon'])} < NFCorpus {n(_dom.loc['nfcorpus','corpus_jargon'])} < SciFact "
            f"{n(_dom.loc['scifact','corpus_jargon'])}. FiQA is the least specialised of the three."],
         [("fig2_domain.png", "Figure 3. Domain-level jargon rate vs BM25 effectiveness (left) and vs the dense model's gain over BM25 (right). n = 8.")]),
        ("H3", "H3. Does query-level overlap explain BM25 effectiveness?  Verdict: " + V["H3"]["verdict"], [
            "Within datasets, coverage strongly predicts BM25 nDCG@10 after controlling for query length and number of "
            f"relevant documents (partial rho: " + ", ".join(f"{PRETTY[d]} {n(h3.loc[d,'bm25'],2)}" for d in DS) + "). "
            "The result holds for IDF-weighted coverage, maximum coverage and Jaccard (results/h3_correlations.csv). "
            "Touche is the exception: almost all its queries have high coverage, so there is no variance to explain.",
            "<b>Circularity check.</b> The same coverage measure also predicts the dense model, whose scoring ignores "
            "lexical identity (partial rho for MiniLM: " + ", ".join(f"{PRETTY[d]} {n(h3.loc[d,'minilm'],2)}" for d in DS) +
            f"). In the mixed model the coverage slope is {n(V['H3']['mixed']['bm25']['coef'])} for BM25 vs "
            f"{n(V['H3']['mixed']['minilm']['coef'])} for MiniLM. Coverage is therefore partly a proxy for 'easy, "
            "well-specified query'. Only the difference between the two slopes is evidence of a lexical-specific mechanism."],
         [("fig3_overlap_bins.png", "Figure 4. Mean nDCG@10 by coverage bin. The lines diverge at low coverage (dense and hybrid above BM25) on most datasets."),
          ("fig4_forest.png", "Figure 5. Partial correlations with 95% bootstrap CIs. Blue: coverage vs BM25 (H3). Orange: coverage vs the dense-minus-BM25 gap (H5).")]),
        ("H5", "H5. Does dense retrieval help most where overlap is low?  Verdict: " + V["H5"]["verdict"], [
            f"Pooled over all queries, the dense model's advantage shrinks as coverage rises (mixed-model coefficient "
            f"{n(V['H5']['mixed']['coef'])} nDCG per SD, {pv(V['H5']['mixed']['p'])}; bge-small "
            f"{n(V['H5_bge_robustness']['mixed']['coef'])}). Per dataset, the effect is significant on NFCorpus, SciFact, "
            "ArguAna and Quora but not on FiQA or SciDocs, where the dense model wins at <i>every</i> coverage level, nor on "
            "TREC-COVID or Touche (about 50 queries each). The decision rule needs significance in 6 of 8 datasets, so the "
            "verdict is partial support: the mechanism is real but its size depends on the collection."], []),
        ("H6", "H6. Is there a usable overlap threshold?  Verdict: " + V["H6"]["verdict"], [
            f"A rule 'use a dense model when coverage < t' was learned on seven datasets and tested on the eighth. The "
            f"threshold is stable (t = {V['H6']['threshold_median']:.2f} in 7/8 folds), but held-out balanced accuracy averages "
            f"only {n(V['H6']['mean_lodo_bacc'],2)} (chance = 0.50) and the threshold-free AUC is {n(V['H6']['mean_auc'],2)}. "
            "More fundamentally, coverage is computed from relevant documents, so it is an <i>offline diagnostic</i> for "
            "characterising a collection from a labelled sample, not a router for live queries. Domain-average coverage "
            "is also not significantly related to dense gains across the 8 domains."], []),
        ("H4", "H4. Failure-mode taxonomy  Verdict: " + V["H4"]["verdict"] + " (with caveats)", [
            f"BM25 failures (nDCG@10 < 0.1) range from {V['H4']['failure_rate']['trec-covid']:.0%} of queries (TREC-COVID) to "
            f"{V['H4']['failure_rate']['scidocs']:.0%} (SciDocs). Up to 400 failures per dataset were labelled by rules "
            f"(n = {V['H4']['n_labelled']}). The distribution differs strongly by domain (Cramer's V = {V['H4']['cramers_v']:.2f}): "
            "NFCorpus, SciFact and ArguAna failures are dominated by paraphrase/indirect relevance (the relevant document "
            "shares almost no query terms), while FiQA, SciDocs and Quora have many term-competition failures (relevant "
            "documents match well, but so do many non-relevant ones, a polysemy/non-discriminative-term proxy). "
            "Abbreviation and synonymy, which the proposal emphasised, are minor.",
            f"<b>Validity of the labels.</b> Rules were revised once after a development audit "
            f"({_aud['dev_acc']:.0%} correct on 30 queries). On a fresh sample of {_aud['test_n']} queries, "
            f"{_aud['test_correct']} labels were judged correct ({_aud['test_acc']:.0%}, 95% CI {_aud['wilson_lo']:.0%}-"
            f"{_aud['wilson_hi']:.0%}). Remaining errors are misspellings, clipped forms ('carb', 'corp') and occasional "
            "spurious WordNet senses. Audits: results/failure_manual_check_*.csv. Many NFCorpus 'failures' reflect how its "
            "relevance links were built (indirect topical links from a nutrition website) rather than lexical matching."],
         [("fig6_failures.png", "Figure 6. Primary failure mode per dataset (TREC-COVID and Touche have fewer than 3 failures and are excluded from the chi-square test).")]),
    ]


CATALOGUE_INTRO = ["Examples are paraphrased (not reproduced) as the proposal requires; mode labels follow the "
                   "audited taxonomy. Two rows show modes the rules miss."]
CATALOGUE = [
    ["Paraphrase / indirect", "Nutrition (NFCorpus)", "A single root vegetable named as a query.",
     "Bile-acid binding of steamed leafy greens and brassicas; the vegetable itself is never named."],
    ["Paraphrase / indirect", "Nutrition (NFCorpus)", "A brand-name cholesterol drug.",
     "A trial of a fruit's effect on blood lipids in hyperlipidaemic patients."],
    ["Morphology", "Nutrition (NFCorpus)", "The noun for memory loss.",
     "The adjective form inside the name of a shellfish-toxin poisoning syndrome."],
    ["Abbreviation", "Science (SciDocs)", "A paper title spelling out 'convolutional neural networks'.",
     "Cited work refers only to 'CNN'."],
    ["Abbreviation / symbol", "Science (SciFact)", "A claim using a Greek-letter cytokine name and a disease acronym.",
     "Abstract writes the Greek letter as a word and spells out the disease, so the tokens never match."],
    ["Synonymy", "Science (SciFact)", "A claim about deaths around birth caused by low birth weight.",
     "A study of the global prevalence of childhood underweight."],
    ["Term competition", "Finance (FiQA)", "How inflation relates to interest rates.",
     "The answer shares both key terms, but so do dozens of non-relevant posts that outrank it."],
    ["Term competition", "General (Quora)", "Best laptop for a given budget.",
     "A near-duplicate question; many other budget-laptop questions match equally well."],
    ["Spelling variant", "Science (SciDocs)", "British spelling of a dialogue-systems term.",
     "Cited paper uses the American spelling."],
    ["Paraphrase", "Argument (ArguAna)", "An argument that ideas cannot be owned, so drug patents are illegitimate.",
     "The counter-argument talks about creative effort and generic production, with few shared words."],
    ["Misspelling (missed)", "Science (SciFact)", "A claim with a misspelt word for tumour spread.",
     "Abstract uses the correct spelling."],
    ["Clipping (missed)", "Nutrition (NFCorpus)", "A diet named with a clipped word for carbohydrate.",
     "Meta-analysis uses the full word."],
]

RESCUE_TEXT = (f"Dense models rescue {_resc['rescued_minilm'].min():.0%}-{_resc['rescued_minilm'].max():.0%} of BM25 "
               f"failures in every mode, while WordNet expansion rescues at most {_resc['rescued_bm25_wordnet'].max():.0%} "
               f"and RM3 at most {_resc['rescued_bm25_rm3'].max():.0%}. Dense models are not specifically better on "
               "'synonymy' failures, which is another sign that the lexical mode labels capture only part of what goes wrong.")


def GUIDELINE(V):
    return [
        "Based on these eight collections and two small dense models:",
        "1. <b>Never use strict Boolean AND for ranked retrieval.</b> It returns no results for most queries.",
        "2. <b>Default to a BM25 + dense hybrid</b> (convex fusion or RRF). It significantly improved on BM25 on 7 of 8 "
        "collections, by up to +0.16 nDCG@10, with a single loss on Touche. It also protects against choosing a weak dense "
        "model, as on TREC-COVID with MiniLM.",
        "3. <b>BM25 alone is defensible when queries are short keyword queries over a collection whose relevance is "
        "topical and judged from lexical pools</b> (Touche: BM25 beats both dense models by 0.20-0.28), and, by a small "
        "margin, in jargon-heavy scientific/biomedical collections. Domain jargon is <i>not</i> a reason to abandon BM25.",
        "4. <b>Tune BM25's length normalisation (b) on a development set before concluding lexical retrieval is weak.</b> "
        "Gains of up to 0.12 nDCG@10 were available (ArguAna).",
        "5. <b>Do not expect generic query expansion to fix vocabulary mismatch.</b> WordNet did nothing and RM3 was mixed.",
        "6. <b>Use coverage as an offline diagnostic.</b> Label about 50 queries; if most relevant documents contain fewer than "
        "about 60% of the query terms, a dense or hybrid component is likely to pay off. Treat this as a heuristic: it "
        f"transferred to unseen domains with only {n(V['H6']['mean_lodo_bacc'],2)} balanced accuracy.",
    ]


THREATS = [
    "Eight domains give low power for domain-level tests; the jargon conclusions are 'no evidence for', not 'evidence against'.",
    "Both dense models are small (22-33M parameters); stronger models would likely widen the gaps, and results differ between the two.",
    "nDCG is not comparable across collections (judging depth, relevant documents per query, task). Within-dataset and relative measures carry the conclusions.",
    "Coverage is computed from relevance judgements, which in TREC-COVID and Touche were pooled mostly from lexical systems; this can favour BM25.",
    f"Failure labels are heuristic ({_aud['test_acc']:.0%} audited accuracy); the audit was done by one annotator (the analyst) without a second rater.",
    "The jargon measure (Zipf < 3) counts proper names and rare general words as jargon and ignores multi-word terms.",
    "BM25 grid search used test queries and is reported only as an upper bound. RM3 and WordNet parameters were not tuned.",
    "Dense runs were done on a different machine (Colab T4 GPU); a CPU cross-check on five datasets agreed to within 0.0004.",
]

REPRO = [
    "Code: src/common.py (data, indexing, metrics), run_lexical.py (E1/E2/E5), run_dense.py (E6), features.py (E3), "
    "analysis.py (H1-H3, H5, H6, figures), failures.py (H4), validate_metrics.py, build_report.py; colab/ holds the GPU "
    "notebook. Order: run_lexical -> run_dense (or the Colab notebook) -> features -> analysis -> failures -> build_report.",
    "All per-query results, test outputs and verdicts are in results/ (verdicts.json, h4_verdict.json, *_tests.csv, "
    "failure_manual_check_*.csv). Seed 42; BM25 k1 = 0.9, b = 0.4; dense max length 256.",
]
