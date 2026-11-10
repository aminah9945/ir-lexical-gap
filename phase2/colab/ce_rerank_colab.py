"""Phase 2 / Agent 4 (E2, E3 at full scale): cross-encoder logits for every evaluated query of the 8 BEIR datasets.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2, max_length 512, fp16 on GPU.
Pool per query = BM25 top-100 (Phase-1 BM25: stem+stop, k1=0.9, b=0.4, identical-id docs dropped)
                 U  H_0.5(D) top-100  U  fused top-10 of H_alpha(D) for alpha in {0, 0.1, ..., 1},
for every dense model D whose Colab run (colab_runs/run_<D>_<ds>.npz, from phase2_dense_runs.ipynb) is on Drive.
Output: <OUT>/ce/ce_<ds>.pkl = {"sample": [all qids], "scores": {(qid, docid): logit}} -- the same format as
phase2/04_improvements/ce_score.py, read automatically by 04_improvements/ce_eval.py. Resumable per dataset.
"""
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASETS, Index, bm25_matrix, query_matrix, search, to_run, load_dataset)  # noqa: E402

OUT = os.environ.get("OUT", "/content/drive/MyDrive/IR_Lexical_Gap/colab_runs")
CE_OUT = os.path.join(OUT, "ce")
os.makedirs(CE_OUT, exist_ok=True)
TOPK = 100
ALPHAS = np.round(np.linspace(0, 1, 11), 1)


def bm25_run(name):
    idx = Index(name)
    qids = list(idx.qrels)
    C, vocab, an = idx.variant(True, True)
    Q = query_matrix([idx.queries[q] for q in qids], vocab, an)
    W, _ = bm25_matrix(C)
    return to_run(search(W, Q, k=1000), qids, idx.doc_ids)


def dense_run(m, name, doc_ids):
    p = os.path.join(OUT, f"run_{m}_{name}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return {str(q): [(doc_ids[i], float(s)) for i, s in zip(ix, ss)] for q, ix, ss in zip(z["qids"], z["idx"], z["scores"])}


def fuse(lb, ld, alphas):
    """Same as common2.convex / run_dense.convex (min-max, absent = 0, stable order B then D)."""
    order, idx, parts = [], {}, []
    for lst in (lb, ld):
        v = np.array([s for _, s in lst], float)
        if len(v):
            lo, hi = v.min(), v.max()
            v = (v - lo) / (hi - lo) if hi > lo else np.ones_like(v)
        parts.append(v)
        for d, _ in lst:
            if d not in idx:
                idx[d] = len(order)
                order.append(d)
    sb, sd = np.zeros(len(order)), np.zeros(len(order))
    for (d, _), v in zip(lb, parts[0]):
        sb[idx[d]] += v
    for (d, _), v in zip(ld, parts[1]):
        sd[idx[d]] += v
    order = np.array(order, dtype=object)
    return {float(a): list(order[np.argsort(-(a * sb + (1 - a) * sd), kind="stable")[:TOPK]]) for a in alphas}


def main():
    from sentence_transformers import CrossEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device=dev)
    if dev == "cuda":
        ce.model.half()
    for name in (sys.argv[1:] or DATASETS):
        out = os.path.join(CE_OUT, f"ce_{name}.pkl")
        doc_ids, texts, queries, qrels = load_dataset(name)
        qids = sorted(qrels)
        if name == "quora":  # 10k queries x ~200 pairs would be ~2M pairs and a ~150 MB file: subsample (stated)
            qn = int(os.environ.get("QUORA_N", "2000"))
            r1, r2 = np.random.default_rng(42), np.random.default_rng(42)
            qids = sorted(set(r1.choice(qids, 150, replace=False).tolist()) |  # = E2 tuning sample S_d
                          set(r2.choice(qids, qn, replace=False).tolist()))
        state = pickle.load(open(out, "rb")) if os.path.exists(out) else {"sample": qids, "scores": {}}
        B = bm25_run(name)
        pool = {q: set(d for d, _ in B.get(q, [])[:TOPK]) for q in qids}
        for m in ("minilm", "bge", "bgebase"):
            D = dense_run(m, name, doc_ids)
            if D is None:
                continue
            for q in qids:
                f = fuse(B.get(q, [])[:1000], D.get(q, [])[:1000], ALPHAS)
                pool[q].update(f[0.5])
                for a in ALPHAS:
                    pool[q].update(f[float(a)][:10])
        pos = {d: i for i, d in enumerate(doc_ids)}
        todo = [(q, d) for q in qids for d in sorted(pool[q]) if (q, d) not in state["scores"] and d in pos]
        todo.sort(key=lambda x: len(texts[pos[x[1]]]) + len(queries[x[0]]))
        print(f"{name}: {len(qids)} queries, {len(todo)} pairs to score", flush=True)
        t0, CH = time.time(), 50000
        for s in range(0, len(todo), CH):
            chunk = todo[s:s + CH]
            sc = ce.predict([(queries[q], texts[pos[d]]) for q, d in chunk], batch_size=128, show_progress_bar=False)
            state["scores"].update({k: float(v) for k, v in zip(chunk, sc)})
            pickle.dump(state, open(out + ".tmp", "wb"))
            os.replace(out + ".tmp", out)
            print(f"  {name}: {s+len(chunk)}/{len(todo)} ({(s+len(chunk))/(time.time()-t0):.0f} pairs/s)", flush=True)
        del B


if __name__ == "__main__":
    main()
