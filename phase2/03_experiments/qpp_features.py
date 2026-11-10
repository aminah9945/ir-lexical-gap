"""W1/W2 features: query-time (qrel-free) predictors.

Tier A (per dataset):   pre-retrieval predictors + post-retrieval predictors on the BM25 top-1000 list.
Tier B (per dataset, per dense model D with a ranking on disk): post-retrieval predictors on D's list,
                        B/D agreement features and the pseudo-coverage proxies pcov (W2).
Formulas follow phase2/02_verification.md Sec. 3 (W1/W2); implementation choices are documented inline.

Outputs results/qpp/tierA_<ds>.csv and results/qpp/tierB_<model>_<ds>.csv (one row per query of the qrels).
Usage: python qpp_features.py [datasets...]     (FORCE=1 to recompute existing files)
No qrels are read except to enumerate the evaluated query ids (the same ids as the Phase-1 per-query CSVs).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import common2 as c2
from common2 import common

sys.path.insert(0, os.path.join(c2.ROOT, "src"))
from features import jargon_rate  # noqa: E402  (Phase-1 jargon definition, Zipf<3)

QDIR = os.path.join(c2.RES2, "qpp")
os.makedirs(QDIR, exist_ok=True)
FORCE = os.environ.get("FORCE") == "1"
MU_DIR = 1000.0      # Dirichlet mu for clarity (implementation choice)
CLAR_DOCS, CLAR_TERMS = 100, 100


# ----------------------------------------------------------------------------- score-list predictors
def score_feats(s, S_C, nq, prefix, shift=False):
    """s: scores sorted desc (top-k list, k<=1000); S_C: corpus-score proxy (mu_1000); nq: |q|."""
    f = {}
    s = np.asarray(s, float)
    if len(s) == 0:
        return {prefix + k: np.nan for k in ["wig5", "nqc100", "smv100", "s1", "s1_s10", "s1_s10_rel", "mu10_mu100"]}
    nq = max(nq, 1)
    k5, k100 = s[:5], s[:100]
    f["wig5"] = float(np.mean(k5 - S_C) / np.sqrt(nq))
    mu100 = k100.mean()
    f["nqc100"] = float(np.sqrt(np.mean((k100 - mu100) ** 2)) / abs(S_C)) if S_C != 0 else np.nan
    if shift:  # cosine scores: s' = s - min(list) + 1e-6 (min over top-1000; top-100 for Quora npz)
        sh = s - s.min() + 1e-6
        k100s, SCs = sh[:100], sh.mean()
    else:
        k100s, SCs = k100, S_C
    m = k100s.mean()
    f["smv100"] = float(np.mean(k100s * np.abs(np.log(k100s / m))) / SCs) if SCs > 0 and np.all(k100s > 0) else np.nan
    s10 = s[min(9, len(s) - 1)]
    f["s1"] = float(s[0])
    f["s1_s10"] = float(s[0] - s10)
    f["s1_s10_rel"] = float((s[0] - s10) / s[0]) if s[0] != 0 else np.nan
    f["mu10_mu100"] = float(s[:10].mean() / mu100) if mu100 != 0 else np.nan
    return {prefix + k: v for k, v in f.items()}


def clarity(rows, scores, C, dl, pc):
    """Clarity (Cronen-Townsend et al. 2002), Meng et al. 2023 variant: P(w|R)=sum_d P(w|d)P(d|q), P(d|q) =
    sum-normalised retrieval scores of the top-100 docs, P(w|d) Dirichlet-smoothed (mu=1000); R clipped to its
    top-100 terms and renormalised; KL(P(.|R) || P(.|C)) in bits. Only terms occurring in the top docs are
    candidates (smoothing mass on unseen terms is dropped)."""
    if len(rows) == 0:
        return np.nan
    sc = np.asarray(scores, float)
    sc = sc - min(sc.min(), 0) + 1e-9
    pdq = sc / sc.sum()
    sub = C[rows]
    U, inv = np.unique(sub.indices, return_inverse=True)
    M = np.zeros((len(rows), len(U)))
    rr = np.repeat(np.arange(len(rows)), np.diff(sub.indptr))
    np.add.at(M, (rr, inv), sub.data)
    pw = pc[U]
    Pwd = (M + MU_DIR * pw[None, :]) / (dl[rows][:, None] + MU_DIR)
    PwR = pdq @ Pwd
    top = np.argsort(-PwR)[:CLAR_TERMS]
    p = PwR[top] / PwR[top].sum()
    return float(np.sum(p * np.log2(p / pw[top])))


def rbo_ext(a, b, p=0.9, k=100):
    a, b = a[:k], b[:k]
    k = min(len(a), len(b))
    if k == 0:
        return np.nan
    sa, sb, x, acc = set(), set(), 0, 0.0
    for d in range(1, k + 1):
        ea, eb = a[d - 1], b[d - 1]
        if ea == eb:
            x += 1
        else:
            x += (ea in sb) + (eb in sa)
        sa.add(ea)
        sb.add(eb)
        acc += (x / d) * p ** d
    return float((x / k) * p ** k + (1 - p) / p * acc)


def pcov(T, rows, C):
    """pcov@10 = mean over top-10 docs of |T & terms(d)|/|T|; pcov_any = |T & union terms(top10)|/|T|."""
    if not T or len(rows) == 0:
        return np.nan, np.nan
    Ts = set(T)
    per, union = [], set()
    for r in rows[:10]:
        ds = set(C.indices[C.indptr[r]:C.indptr[r + 1]].tolist()) & Ts
        per.append(len(ds) / len(Ts))
        union |= ds
    return float(np.mean(per)), len(union) / len(Ts)


# ----------------------------------------------------------------------------- main
def main(ds):
    outA = os.path.join(QDIR, f"tierA_{ds}.csv")
    models = [m for m in c2.DENSE_MODELS if c2.has_run(ds, m)]
    todoB = [m for m in models if FORCE or not os.path.exists(os.path.join(QDIR, f"tierB_{m}_{ds}.csv"))]
    if not FORCE and os.path.exists(outA) and not todoB:
        print("features exist", ds)
        return
    t0 = time.time()
    idx = common.Index(ds)
    qids = list(idx.qrels)                       # evaluated query ids (no relevance information used)
    qtext = {q: idx.queries[q] for q in qids}
    doc_ids = idx.doc_ids
    C, vocab, an = idx.variant(True, True)
    del idx
    C = C.tocsr()
    N = C.shape[0]
    df = np.bincount(C.indices, minlength=C.shape[1]).astype(float)
    cf = np.asarray(C.sum(0)).ravel().astype(float)
    dl = np.asarray(C.sum(1)).ravel().astype(float)
    T_C = cf.sum()
    pc = cf / T_C
    # VAR(t): std over docs containing t of w = 1 + ln(tf) * ln(1+N/df)  (= ln(1+N/df) * std(ln tf))
    L = C.copy().astype(float)
    L.data = np.log(L.data)
    s1 = np.asarray(L.sum(0)).ravel()
    L.data **= 2
    s2 = np.asarray(L.sum(0)).ravel()
    del L
    with np.errstate(invalid="ignore", divide="ignore"):
        var_ln = np.maximum(s2 / df - (s1 / df) ** 2, 0)
        VAR = np.log(1 + N / df) * np.sqrt(var_ln)
        IDF = np.log(N / df)
        ICTF = np.log2(T_C / cf)
        SCQ = (1 + np.log(cf)) * np.log(1 + N / df)
    row_of = {d: i for i, d in enumerate(doc_ids)}

    bm25, bmu = c2.load_run(ds, "bm25", with_mu=True)
    doA = FORCE or not os.path.exists(outA)
    qterms, rowsA = {}, []
    for q in qids:
        raw = common.tokenize(qtext[q])
        toks = an(qtext[q])
        inv = [vocab[t] for t in toks if t in vocab]
        T = sorted(set(inv))
        qterms[q] = T
        if not doA:
            continue
        n_oov = len({t for t in toks if t not in vocab})
        f = dict(dataset=ds, qid=q, q_len=len(raw), q_terms=len(T), q_oov_frac=n_oov / max(1, n_oov + len(T)),
                 q_jargon=jargon_rate(raw))
        if T:
            Ta = np.array(T)
            f.update(idf_mean=IDF[Ta].mean(), idf_max=IDF[Ta].max(), idf_min=IDF[Ta].min(), idf_std=IDF[Ta].std(),
                     avictf=ICTF[Ta].mean(), scq_sum=SCQ[Ta].sum(), scq_mean=SCQ[Ta].mean(), scq_max=SCQ[Ta].max(),
                     var_sum=np.nansum(VAR[Ta]), var_max=np.nanmax(VAR[Ta]))
            # SCS over analysed query tokens (with repetitions), P_ml(t|q) = count/|q|
            cnt = pd.Series(inv).value_counts()
            pq = cnt.values / cnt.values.sum()
            f["scs"] = float(np.sum(pq * np.log2(pq / pc[cnt.index.values])))
        lst = bm25.get(q, [])
        sc = [s for _, s in lst]
        f.update(score_feats(sc, bmu.get(q, np.nan), len(T), "B_"))
        rows = np.array([row_of[d] for d, _ in lst[:CLAR_DOCS]], dtype=int)
        f["B_clarity100"] = clarity(rows, sc[:CLAR_DOCS], C, dl, pc)
        rowsA.append(f)
    if doA:
        A = pd.DataFrame(rowsA)
        c2.assert_no_qrel_features(A.columns)
        A.to_csv(outA, index=False)
        print(f"tierA {ds}: {len(A)} queries ({time.time()-t0:.0f}s)", flush=True)

    for m in todoB:
        t1 = time.time()
        drun, dmu = c2.load_run(ds, m, with_mu=True)
        hyb = c2.convex(bm25, drun)
        rowsB = []
        for q in qids:
            T = qterms[q]
            bl, dlst = bm25.get(q, []), drun.get(q, [])
            bdoc, ddoc = [d for d, _ in bl], [d for d, _ in dlst]
            dsc = [s for _, s in dlst]
            f = dict(dataset=ds, qid=q)
            f.update(score_feats(dsc, dmu.get(q, np.nan), len(T), "D_", shift=True))
            drows = np.array([row_of[d] for d in ddoc[:CLAR_DOCS]], dtype=int)
            f["D_clarity100"] = clarity(drows, dsc[:CLAR_DOCS], C, dl, pc)
            b10, d10, b100, d100 = set(bdoc[:10]), set(ddoc[:10]), set(bdoc[:100]), set(ddoc[:100])
            f["jacc10"] = len(b10 & d10) / max(1, len(b10 | d10))
            f["jacc100"] = len(b100 & d100) / max(1, len(b100 | d100))
            f["rbo100"] = rbo_ext(bdoc, ddoc, 0.9, 100)
            dpos = {d: i + 1 for i, d in enumerate(ddoc)}
            bpos = {d: i + 1 for i, d in enumerate(bdoc)}
            f["rank_Btop1_in_D"] = dpos.get(bdoc[0], 1001) if bdoc else 1001
            f["rank_Dtop1_in_B"] = bpos.get(ddoc[0], 1001) if ddoc else 1001
            hs = dict(hyb.get(q, []))
            den = sum(hs.get(d, 0) for d in (b10 | d10))
            f["fused_overlap10"] = sum(hs.get(d, 0) for d in (b10 & d10)) / den if den > 0 else np.nan
            brows = np.array([row_of[d] for d in bdoc[:10]], dtype=int)
            f["pcov_B10"], f["pcov_any_B10"] = pcov(T, brows, C)
            f["pcov_D10"], f["pcov_any_D10"] = pcov(T, drows, C)
            f["pcov_gap"] = f["pcov_B10"] - f["pcov_D10"] if T else np.nan
            rowsB.append(f)
        B = pd.DataFrame(rowsB)
        c2.assert_no_qrel_features(B.columns)
        B.to_csv(os.path.join(QDIR, f"tierB_{m}_{ds}.csv"), index=False)
        print(f"tierB {m} {ds}: {len(B)} queries ({time.time()-t1:.0f}s)", flush=True)
        del drun, hyb


if __name__ == "__main__":
    for d in (sys.argv[1:] or c2.ALL_DATASETS):
        main(d)
