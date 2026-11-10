"""Cross-check our metric implementation against pytrec_eval on the BM25 runs."""
import sys, numpy as np, pytrec_eval
from common import load_dataset, load_run, evaluate_run, CORE
for d in sys.argv[1:] or CORE:
    _, _, _, qrels = load_dataset(d)
    run = load_run(d, "bm25")
    ours = evaluate_run(run, qrels)
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100", "map_cut.100"})
    ref = ev.evaluate({q: {doc: s for doc, s in r} for q, r in run.items()})
    for a, b in [("ndcg@10", "ndcg_cut_10"), ("recall@100", "recall_100"), ("map@100", "map_cut_100")]:
        diff = max(abs(ours[q][a] - ref[q][b]) for q in ref)
        print(f"{d:10s} {a:11s} ours={np.mean([ours[q][a] for q in ref]):.4f} pytrec={np.mean([ref[q][b] for q in ref]):.4f} max|diff|={diff:.2e}")
