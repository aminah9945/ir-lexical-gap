"""E7 / H4: heuristic failure-mode taxonomy for BM25 failures (nDCG@10 < 0.1).

Labels (multi-label; primary label by priority ABBREV > MORPH > SYN > AMBIG > PARAPHRASE > OTHER):
  ABBREV     acronym<->expansion mismatch between query and relevant doc
  MORPH      query word missing but a relevant doc has a word sharing a long prefix (>=5 chars and
             >=70% of the shorter word) that the stemmer did not conflate (e.g. 'diabetes'/'diabetic')
  SYN        a WordNet synonym (first two senses, >=4 chars) of a missing content word
             (Zipf < 5) occurs in a relevant document
  AMBIG      polysemy/term-competition proxy: best relevant doc contains >=50% of query terms, yet
             top-10 non-relevant docs match the query at least as well as it does
  PARAPHRASE relevant docs contain < 50% of query terms and no rule above explains it
  OTHER      everything else
These are heuristics. Rules were revised once after a development audit
(results/failure_manual_check_dev.csv); accuracy is then measured on a fresh sample
(results/failure_manual_check_test.csv).
H4 VERDICT RULE: SUPPORTED if >= 70% of failures receive a non-OTHER label AND the primary-label
distribution differs across datasets (chi-square p < .05); PARTIALLY if only one holds.
"""
import json
import os
import re

import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from scipy import stats
from wordfreq import zipf_frequency
import os.path


def _cp(a, b):
    return os.path.commonprefix([a, b])

from common import (DATASETS, RES, SEED, FIG, PRETTY, LUCENE_STOP, load_dataset, tokenize, _stemmer)

FAIL_T = 0.1
MAX_PER_DS = 400
ACR = re.compile(r"\b[A-Z][A-Z0-9]{1,5}s?\b")
PRIORITY = ["AMBIG", "ABBREV", "MORPH", "SYN", "PARAPHRASE", "OTHER"]
_syn = {}


def synonyms(w):
    if w not in _syn:
        s = set()
        for ss in wn.synsets(w)[:2]:
            for l in ss.lemma_names():
                l = l.lower()
                if "_" not in l and l != w:
                    s.add(_stemmer.stemWord(l))
        _syn[w] = s
    return _syn[w]


def initials_runs(words, maxlen=6):
    out = set()
    for i in range(len(words)):
        for j in range(i + 2, min(len(words), i + maxlen) + 1):
            out.add("".join(w[0] for w in words[i:j]))
    return out


def label_query(qtext, rel_texts, feat):
    qraw = [t for t in tokenize(qtext) if t not in LUCENE_STOP]
    qstem = _stemmer.stemWords(qraw)
    rel_tok = [tokenize(t) for t in rel_texts]
    rel_stem = set()
    rel_raw = set()
    for toks in rel_tok:
        rel_raw.update(toks)
        rel_stem.update(_stemmer.stemWords(toks))
    missing_all = list(dict.fromkeys((r, s) for r, s in zip(qraw, qstem) if s not in rel_stem and not r.isdigit()))
    # only the 5 most specific missing words (lowest general-English frequency) are examined,
    # so long queries (ArguAna ~200 tokens) do not trigger rules by chance
    missing = sorted(missing_all, key=lambda x: zipf_frequency(x[0], "en"))[:5]
    labels, ev = set(), []
    # AMBIG first: if relevant docs already contain most query terms, the failure is not a
    # vocabulary mismatch but term competition (polysemy / non-discriminative terms)
    if feat["cov_max"] >= 0.5 and feat["nonrel_top10_cov"] >= feat["cov_max"]:
        labels.add("AMBIG"); ev.append(f"rel cov_max={feat['cov_max']:.2f}, top-10 nonrel cov={feat['nonrel_top10_cov']:.2f}")
    # ABBREV: require acronym<->expansion evidence, not merely an absent acronym
    q_acr = [a.rstrip("s") for a in ACR.findall(qtext) if not a.isdigit() and len(a.rstrip("s")) >= 3]
    q_acr = [a for a in q_acr if _stemmer.stemWord(a.lower()) not in rel_stem]
    qinit = initials_runs([w for w in qraw if w.isalpha()])
    for t in rel_texts:
        for a in set(ACR.findall(t)):
            al = a.lower().rstrip("s")
            if len(al) >= 3 and al in qinit and al not in qraw:
                labels.add("ABBREV"); ev.append(f"relevant doc uses acronym '{a}' for a query phrase")
                break
        if "ABBREV" in labels:
            break
    if "ABBREV" not in labels and q_acr:
        for t in rel_texts:
            dinit = initials_runs([w for w in tokenize(t) if w.isalpha() and w not in LUCENE_STOP][:400])
            hit = [a for a in q_acr if a.lower() in dinit]
            if hit:
                labels.add("ABBREV"); ev.append(f"query acronym '{hit[0]}' is spelled out in relevant doc")
                break
    # MORPH and SYN on the most specific missing words
    for r, s in missing:
        if len(r) >= 5:
            m = [w for w in rel_raw if len(w) >= 5 and _stemmer.stemWord(w) != s
                 and len(_cp(r, w)) >= max(5, int(0.7 * min(len(r), len(w))))]
            if m:
                labels.add("MORPH"); ev.append(f"'{r}' vs '{m[0]}'")
        if len(r) >= 4 and zipf_frequency(r, "en") < 5.0:  # content words only
            syn = {x for x in synonyms(r) & rel_stem if len(x) >= 4}
            if syn:
                labels.add("SYN"); ev.append(f"'{r}' ~ '{sorted(syn)[0]}'")
    if feat["cov_max"] < 0.5 and not labels:
        labels.add("PARAPHRASE"); ev.append(f"rel cov_max={feat['cov_max']:.2f}")
    if not labels:
        labels.add("OTHER")
    primary = next(p for p in PRIORITY if p in labels)
    return primary, sorted(labels), "; ".join(ev[:3]), missing


def main():
    rng = np.random.default_rng(SEED)
    pq = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in os.listdir(RES)
                    if f.startswith("perquery_") and f.endswith(".csv")])
    pq["qid"] = pq.qid.astype(str)
    W = pq.pivot_table(index=["dataset", "qid"], columns="system", values="ndcg@10").reset_index()
    rows = []
    for d in DATASETS:
        feats = pd.read_csv(os.path.join(RES, f"features_{d}.csv"))
        feats["qid"] = feats.qid.astype(str)
        feats = feats.set_index("qid")
        doc_ids, texts, queries, qrels = load_dataset(d)
        did = {x: i for i, x in enumerate(doc_ids)}
        w = W[W.dataset == d].set_index("qid")
        fails = [q for q in w[w.bm25 < FAIL_T].index if q in feats.index]
        n_fail_total = len(fails)
        if len(fails) > MAX_PER_DS:
            fails = sorted(rng.choice(fails, MAX_PER_DS, replace=False).tolist())
        for q in fails:
            pos = [x for x, g in qrels[q].items() if g > 0 and x in did]
            # cap: use up to 20 highest-graded relevant docs
            pos = sorted(pos, key=lambda x: -qrels[q][x])[:20]
            prim, labs, ev, missing = label_query(queries[q], [texts[did[x]] for x in pos], feats.loc[q])
            r = dict(dataset=d, qid=q, n_fail_total=n_fail_total, primary=prim, labels="|".join(labs),
                     evidence=ev, missing_terms=",".join(m[0] for m in missing[:8]),
                     cov_max=feats.loc[q, "cov_max"], query=queries[q],
                     rel_snippet=texts[did[pos[0]]][:300] if pos else "")
            for s in ["bm25", "minilm", "bge", "hybrid_rrf_minilm", "bm25_rm3", "bm25_wordnet"]:
                if s in w:
                    r[f"rescued_{s}"] = (w.loc[q, s] >= FAIL_T) if not np.isnan(w.loc[q, s]) else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "failure_labels.csv"), index=False)
    ct = pd.crosstab(df.dataset, df.primary).reindex(index=DATASETS, columns=PRIORITY, fill_value=0)
    ct.to_csv(os.path.join(RES, "failure_crosstab.csv"))
    big = ct[ct.sum(1) >= 20]  # chi-square only on datasets with >= 20 labelled failures
    big = big.loc[:, big.sum(0) > 0]
    chi2, p, dof, _ = stats.chi2_contingency(big)
    n = ct.values.sum()
    n = big.values.sum()
    cv = np.sqrt(chi2 / (n * (min(big.shape) - 1)))
    expl = (df.primary != "OTHER").mean()
    resc = df.groupby("primary")[[c for c in df if c.startswith("rescued_")]].mean().reindex(PRIORITY)
    resc.to_csv(os.path.join(RES, "failure_rescue_rates.csv"))
    fail_rate = W.groupby("dataset").apply(lambda g: (g.bm25 < FAIL_T).mean()).reindex(DATASETS)
    a, b = expl >= 0.7, p < 0.05
    v = "SUPPORTED" if a and b else ("PARTIALLY SUPPORTED" if a or b else "NOT SUPPORTED")
    out = dict(verdict=v, explained_frac=expl, chi2=chi2, p=p, dof=dof, cramers_v=cv,
               failure_rate=fail_rate.to_dict(), n_labelled=len(df))
    with open(os.path.join(RES, "h4_verdict.json"), "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}, f, indent=1)
    print(ct, "\n", out)
    print(resc.round(2))

    # Figure 6: stacked share of primary labels per dataset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"ABBREV": "#2a78d6", "MORPH": "#eb6834", "SYN": "#1baf7a", "AMBIG": "#eda100",
            "PARAPHRASE": "#e87ba4", "OTHER": "#b5b4ae"}
    share = ct.div(ct.sum(1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 3.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    left = np.zeros(len(share))
    for c in PRIORITY:
        ax.barh(range(len(share)), share[c], left=left, color=cols[c], edgecolor="#fcfcfb", lw=2,
                label=c, height=0.7)
        for i, (l, v) in enumerate(zip(left, share[c])):
            if v >= 0.08:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="#1f1f1e")
        left += share[c].values
    ax.set_yticks(range(len(share)), [f"{PRETTY[d]} (n={ct.loc[d].sum()})" for d in share.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of BM25 failures (nDCG@10 < 0.1), primary label")
    ax.legend(ncol=6, fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_failures.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
