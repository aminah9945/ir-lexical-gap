import sys, os
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path[:0]=[os.path.join(HERE,"..","..","03_experiments"),os.path.join(HERE,"..","..","..","src")]
import common2 as c2
B=10000
def macro_boot(diffs, n=B, seed=42):
    rng = np.random.default_rng(seed)
    ds = [np.asarray(v, float) for v in diffs]
    obs = float(np.mean([v.mean() for v in ds])); bs = np.zeros(n)
    for v in ds:
        for s0 in range(0, n, 1000):
            m = min(1000, n - s0); bs[s0:s0 + m] += v[rng.integers(0, len(v), (m, len(v)))].mean(1)
    bs /= len(ds); lo, hi = np.percentile(bs, [2.5, 97.5]); _, p2, p1 = c2.perm_test_macro(ds)
    return dict(delta=obs, lo=float(lo), hi=float(hi), p_one=float(p1), p_two=float(p2), k=len(ds))

