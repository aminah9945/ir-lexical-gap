"""Phase-2 shared utilities (Agent 3).

- paths, dataset discovery from the files that are present
- run loading from Phase-1 pickles (cache/runs/<ds>__<sys>.pkl) or Colab npz (colab_runs/run_<m>_<ds>.npz)
- fixed convex hybrid H(D) = 0.5 minmax(B) + 0.5 minmax(D) (identical to src/run_dense.py::convex)
- statistics: paired randomization test, paired bootstrap CI, Holm, DerSimonian-Laird random-effects meta-analysis
- the qrel-feature whitelist assertion for router / QPP inputs

Phase-1 code (src/common.py) is imported read-only; nothing in src/ or results/ is modified.
"""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
import common  # noqa: E402  (Phase-1 module, read-only)

P1_RES = os.path.join(ROOT, "results")
P1_RUNS = os.path.join(ROOT, "cache", "runs")
RES2 = os.path.join(HERE, "results")
FIG2 = os.path.join(HERE, "figures")
CACHE2 = os.path.join(HERE, "cache")
COLAB = os.environ.get("COLAB_RUNS", os.path.join(HERE, "colab_runs"))
for _d in (RES2, FIG2, CACHE2):
    os.makedirs(_d, exist_ok=True)

ALL_DATASETS = list(common.DATASETS)
PRETTY = dict(common.PRETTY)
SHORT = {"nfcorpus": "NFC", "scifact": "SciF", "fiqa": "FiQA", "scidocs": "SciD", "arguana": "ArgA",
         "trec-covid": "COVID", "quora": "Quora", "webis-touche2020": "Touche"}
SEED = 42
N_PERM = 10000
N_BOOT = 10000
TIE = 0.01
DENSE_MODELS = ["minilm", "bge", "bgebase"]
MODEL_PRETTY = {"minilm": "MiniLM", "bge": "BGE-s", "bgebase": "BGE-b"}

# colours (task spec)
C_BM25, C_DENSE, C_HYB, C_ROUTER = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
C_INK, C_MUTED, C_GRID = "#1f1f1e", "#6b6a66", "#e4e3df"

# qrel-derived columns of results/features_*.csv: forbidden as router / QPP inputs
FORBIDDEN = {"n_rel", "cov_mean", "cov_max", "idfcov_mean", "idfcov_max", "jac_mean", "rel_doc_len",
             "nonrel_top10_cov"}
ALLOWED_P1 = {"q_len", "q_terms", "q_oov_frac", "q_mean_idf", "q_jargon"}


def assert_no_qrel_features(cols):
    bad = FORBIDDEN.intersection(cols)
    bad |= {c for c in cols if any(c.startswith(f) or c.endswith("_" + f) for f in FORBIDDEN)}
    assert not bad, f"qrel-derived features in router input: {sorted(bad)}"


# ----------------------------------------------------------------------------- per-query labels
def _read_pq(path):
    d = pd.read_csv(path, dtype={"qid": str})
    return d


def perquery(dataset, metric="ndcg@10"):
    """Wide per-query table {system: metric} from Phase-1 CSVs plus Colab/Phase-2 CSVs.

    Systems: bm25, tfidf, ... (lexical); minilm, hybrid_cc_minilm, bge, hybrid_cc_bge (Phase 1);
    bgebase / hybrid_cc_bgebase (computed here from Colab runs, see hybrid_perquery_cache);
    bm25lodo (+ hybrid_cc_<m>_bm25lodo) from W6.
    """
    frames = [_read_pq(f) for f in glob.glob(os.path.join(P1_RES, f"perquery_*_{dataset}.csv"))]
    frames += [_read_pq(f) for f in glob.glob(os.path.join(RES2, "perquery", f"perquery_*_{dataset}.csv"))]
    pq = pd.concat(frames)
    pq = pq.drop_duplicates(["system", "qid"], keep="first")  # Phase-1 values take precedence
    return pq.pivot_table(index="qid", columns="system", values=metric)


# ----------------------------------------------------------------------------- runs
_doc_ids_cache = {}


def doc_ids_of(dataset):
    """Corpus order of common.load_dataset(ds)[0] (= Index.doc_ids, same corpus.jsonl order)."""
    if dataset not in _doc_ids_cache:
        p = os.path.join(CACHE2, f"docids_{dataset}.pkl")
        if os.path.exists(p):
            _doc_ids_cache[dataset] = pickle.load(open(p, "rb"))
        else:
            ids = []
            import json
            with open(os.path.join(common.DATA, dataset, "corpus.jsonl")) as f:
                for line in f:
                    ids.append(str(json.loads(line)["_id"]))
            pickle.dump(ids, open(p, "wb"))
            _doc_ids_cache[dataset] = ids
    return _doc_ids_cache[dataset]


def colab_npz(model, dataset):
    return os.path.join(COLAB, f"run_{model}_{dataset}.npz")


def has_run(dataset, system):
    if os.path.exists(os.path.join(P1_RUNS, f"{dataset}__{system}.pkl")):
        return True
    if os.path.exists(os.path.join(CACHE2, "runs", f"{dataset}__{system}.pkl")):
        return True
    return system in DENSE_MODELS and os.path.exists(colab_npz(system, dataset))


def load_run(dataset, system, with_mu=False):
    """Return {qid: [(docid, score), ...]} sorted by score desc, from whichever source exists.

    Order of preference: Phase-1 pickle, Phase-2 pickle (cache/runs), Colab npz.
    with_mu=True additionally returns {qid: mu1000} (mean of top-1000 scores; for pickles the
    mean of the list itself, which has <=1000 entries).
    """
    for p in (os.path.join(P1_RUNS, f"{dataset}__{system}.pkl"), os.path.join(CACHE2, "runs", f"{dataset}__{system}.pkl")):
        if os.path.exists(p):
            with open(p, "rb") as f:
                run = pickle.load(f)
            run = {str(q): v for q, v in run.items()}
            if with_mu:
                mu = {q: (float(np.mean([s for _, s in v])) if v else np.nan) for q, v in run.items()}
                return run, mu
            return run
    p = colab_npz(system, dataset)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        ids = doc_ids_of(dataset)
        qids = [str(q) for q in z["qids"]]
        idx, sc = z["idx"], z["scores"]
        run = {q: [(ids[i], float(s)) for i, s in zip(ix, ss)] for q, ix, ss in zip(qids, idx, sc)}
        if with_mu:
            return run, {q: float(m) for q, m in zip(qids, z["mu1000"])}
        return run
    raise FileNotFoundError(f"no run for {dataset}/{system}")


def save_run2(run, dataset, system):
    d = os.path.join(CACHE2, "runs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{dataset}__{system}.pkl"), "wb") as f:
        pickle.dump(run, f, protocol=4)


def convex(r1, r2, alpha=0.5, depth=1000):
    """Exact copy of src/run_dense.py::convex (importing run_dense would import torch)."""
    fused = {}
    for q in r1:
        sc = {}
        for r, w in ((r1, alpha), (r2, 1 - alpha)):
            lst = r.get(q, [])[:depth]
            if not lst:
                continue
            v = np.array([s for _, s in lst])
            lo, hi = v.min(), v.max()
            for d, s in lst:
                sc[d] = sc.get(d, 0) + w * ((s - lo) / (hi - lo) if hi > lo else 1.0)
        fused[q] = sorted(sc.items(), key=lambda x: -x[1])[:depth]
    return fused


def datasets_with(systems_label=(), runs=()):
    """Datasets for which every per-query label system and every run in `runs` is present."""
    out = []
    for d in ALL_DATASETS:
        ok = all(has_run(d, s) for s in runs)
        if ok and systems_label:
            try:
                cols = set(perquery(d).columns)
            except ValueError:
                cols = set()
            ok = all(s in cols for s in systems_label)
        if ok:
            out.append(d)
    return out


# ----------------------------------------------------------------------------- statistics
def perm_test(d, n=N_PERM, seed=SEED):
    """Two-sided paired randomization (sign-flip) test of mean(d) = 0."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    cnt, done = 0, 0
    while done < n:
        m = min(1000, n - done)
        s = rng.choice([-1.0, 1.0], size=(m, len(d)))
        cnt += int((np.abs((s * d).mean(1)) >= obs - 1e-12).sum())
        done += m
    return (cnt + 1) / (n + 1)


def perm_test_macro(diffs_by_ds, n=N_PERM, seed=SEED):
    """Stratified sign-flip test of the macro mean (mean over datasets of per-dataset mean diffs).
    Returns (macro_diff, p_two_sided, p_one_sided_greater)."""
    rng = np.random.default_rng(seed)
    ds = [np.asarray(v, float) for v in diffs_by_ds]
    obs = np.mean([v.mean() for v in ds])
    null = np.zeros(n)
    for v in ds:
        for s0 in range(0, n, 1000):
            m = min(1000, n - s0)
            null[s0:s0 + m] += (rng.choice([-1.0, 1.0], size=(m, len(v))) * v).mean(1)
    null /= len(ds)
    p2 = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n + 1)
    p1 = (np.sum(null >= obs - 1e-12) + 1) / (n + 1)
    return obs, p2, p1


def boot_ci(d, n=N_BOOT, seed=SEED):
    """Paired bootstrap of the mean difference: (lo, hi, se)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for s0 in range(0, n, 1000):
        m = min(1000, n - s0)
        bs[s0:s0 + m] = d[rng.integers(0, len(d), (m, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return lo, hi, bs.std(ddof=1)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    adj = np.empty_like(pv)
    m, run = len(pv), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pv[i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def dl_meta(est, se):
    """DerSimonian-Laird random-effects pooled estimate. Returns dict(est, se, lo, hi, p, tau2, I2, Q, k)."""
    from scipy import stats
    y = np.asarray(est, float)
    v = np.asarray(se, float) ** 2
    v = np.maximum(v, 1e-12)
    k = len(y)
    w = 1 / v
    yf = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - yf) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c) if k > 1 else 0.0
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    se_mu = float(np.sqrt(1 / np.sum(ws)))
    z = mu / se_mu
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(est=mu, se=se_mu, lo=mu - 1.96 * se_mu, hi=mu + 1.96 * se_mu,
                p=float(2 * stats.norm.sf(abs(z))), tau2=tau2, I2=I2, Q=Q, k=k)


def paired_summary(a, b, tie=TIE, seed=SEED, n_perm=N_PERM, n_boot=N_BOOT):
    """Stats of d = a - b: n, mean, CI, se, d_z, win/tie/loss shares, p (randomization)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi, se = boot_ci(d, n_boot, seed)
    sd = d.std(ddof=1)
    return dict(n=len(d), mean_a=float(np.mean(a)), mean_b=float(np.mean(b)), delta=float(d.mean()),
                ci_lo=lo, ci_hi=hi, se_boot=se, d_z=float(d.mean() / sd) if sd > 0 else 0.0,
                win=float(np.mean(d > tie)), tie=float(np.mean(np.abs(d) <= tie)), loss=float(np.mean(d < -tie)),
                p=perm_test(d, n_perm, seed))


# ----------------------------------------------------------------------------- plotting
def mpl_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
    return plt
