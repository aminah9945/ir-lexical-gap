"""E2/E3: score (query, doc) pairs with cross-encoder/ms-marco-MiniLM-L-6-v2 (max_length 512) and cache the logits.

Pool per sampled query (pre-declared in 04_interpretation.md):
  BM25 top-100  U  H_0.5(D) top-100  U  fused top-10 of every alpha-grid list (for each dense model D with runs).
Sample S_d = min(n_d, N_SAMPLE) evaluated queries per dataset, drawn with seed 42 from the sorted qrel query ids
(N_SAMPLE=150 by default; N_SAMPLE=0 means all queries, used by the Colab notebook).
Pairs are scored sorted by text length (less padding); results are saved per dataset in chunks (resumable).

Usage: python ce_score.py [ds ...]      env: N_SAMPLE, CE_THREADS (default 3), CE_BATCH (default 32)
Output: cache/ce_<ds>.pkl  {"sample": [qids], "scores": {(qid, docid): logit}}
"""
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03_experiments"))
import common2 as c2  # noqa: E402

CACHE4 = os.path.join(HERE, "cache")
CE_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
N_SAMPLE = int(os.environ.get("N_SAMPLE", "150"))
TOPK = 100
# cheapest first, so partial results are usable early
ORDER = ["scifact", "nfcorpus", "quora", "webis-touche2020", "trec-covid", "fiqa", "scidocs", "arguana"]


def sample_qids(qrels, n=N_SAMPLE, seed=c2.SEED):
    qs = sorted(qrels)
    if n <= 0 or n >= len(qs):
        return qs
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(qs, n, replace=False).tolist())


def hybrid_top(ds, model, qids):
    p = os.path.join(c2.P1_RUNS, f"{ds}__hybrid_cc_{model}.pkl")
    if os.path.exists(p):
        H = c2.load_run(ds, f"hybrid_cc_{model}")
        return {q: [d for d, _ in H.get(q, [])[:TOPK]] for q in qids}
    B, D = c2.load_run(ds, "bm25"), c2.load_run(ds, model)
    sub = c2.convex({q: B.get(q, []) for q in qids}, D)
    return {q: [d for d, _ in sub.get(q, [])[:TOPK]] for q in qids}


def pool_for(ds, qids):
    B = c2.load_run(ds, "bm25")
    pool = {q: set(d for d, _ in B.get(q, [])[:TOPK]) for q in qids}
    for m in c2.DENSE_MODELS:
        if not c2.has_run(ds, m):
            continue
        for q, lst in hybrid_top(ds, m, qids).items():
            pool[q].update(lst)
        f = os.path.join(CACHE4, f"fused_top10_{m}_{ds}.pkl")
        if os.path.exists(f):
            t10 = pickle.load(open(f, "rb"))
            for a, dct in t10.items():
                for q in qids:
                    pool[q].update(dct.get(q, []))
    return pool


def main(dss):
    import torch
    torch.set_num_threads(int(os.environ.get("CE_THREADS", "3")))
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CE_NAME, max_length=512, device="cuda" if torch.cuda.is_available() else "cpu")
    bs = int(os.environ.get("CE_BATCH", "32"))
    for ds in dss:
        out = os.path.join(CACHE4, f"ce_{ds}.pkl")
        doc_ids, texts, queries, qrels = c2.common.load_dataset(ds)
        qids = sample_qids(qrels)
        state = pickle.load(open(out, "rb")) if os.path.exists(out) else {"sample": qids, "scores": {}}
        assert state["sample"] == qids, "sample changed; delete the cache file"
        pool = pool_for(ds, qids)
        pos = {d: i for i, d in enumerate(doc_ids)}
        todo = [(q, d) for q in qids for d in sorted(pool[q]) if (q, d) not in state["scores"] and d in pos]
        print(f"{ds}: {len(qids)} queries, {sum(len(v) for v in pool.values())} pairs, {len(todo)} to score",
              flush=True)
        todo.sort(key=lambda x: len(texts[pos[x[1]]]) + len(queries[x[0]]))
        t0, CH = time.time(), 2000
        for s in range(0, len(todo), CH):
            chunk = todo[s:s + CH]
            sc = ce.predict([(queries[q], texts[pos[d]]) for q, d in chunk], batch_size=bs,
                            show_progress_bar=False)
            state["scores"].update({k: float(v) for k, v in zip(chunk, sc)})
            pickle.dump(state, open(out + ".tmp", "wb"))
            os.replace(out + ".tmp", out)
            el = time.time() - t0
            print(f"  {ds}: {s+len(chunk)}/{len(todo)} pairs, {(s+len(chunk))/el:.1f} pairs/s", flush=True)
        print(f"{ds} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)
