"""LAPTOP KIT version (fp16, batch 32, adds bgelarge). Phase 2 (v2): dense rankings for MiniLM, bge-small, bge-base on 8 BEIR datasets.
Saves per (model, dataset): run_<m>_<d>.npz with qids, doc indices (into corpus order of load_dataset) and float32
scores for the top-K (K=1000; Quora K=100 to keep files small) after removing the doc whose id equals the query id,
plus mu1000 = mean of the top-1000 scores per query (corpus-score proxy for QPP, Meng et al. 2023), and the
per-query metric CSV. Resumable: finished (model, dataset) pairs are skipped."""
import gzip, os, pickle, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, load_dataset, evaluate_run, per_query_rows
from sentence_transformers import SentenceTransformer

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"))
os.makedirs(OUT, exist_ok=True)
Q_PREFIX = "Represent this sentence for searching relevant passages: "
MODELS = {"minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
          "bge": ("BAAI/bge-small-en-v1.5", Q_PREFIX),
          "bgebase": ("BAAI/bge-base-en-v1.5", Q_PREFIX),
          "bgelarge": ("BAAI/bge-large-en-v1.5", Q_PREFIX)}
K_DEFAULT, K_QUORA = 1000, 100
BS = int(os.environ.get("BS", "32"))  # laptop kit: small batches for 6 GB VRAM
dev = "cuda" if torch.cuda.is_available() else "cpu"
for mkey in (sys.argv[1:] or list(MODELS)):
    name, prefix = MODELS[mkey]
    model = SentenceTransformer(name, device=dev)
    if dev == "cuda": model.half()  # laptop kit: fp16 to fit 6 GB VRAM
    print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    model.max_seq_length = 256
    for d in DATASETS:
        runf = os.path.join(OUT, f"run_{mkey}_{d}.npz")
        if os.path.exists(runf):
            print("skip", mkey, d); continue
        t = time.time()
        doc_ids, texts, queries, qrels = load_dataset(d)
        qids = list(qrels)
        pre = prefix if d != "quora" else ""
        D = model.encode(texts, batch_size=BS, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        Q = model.encode([pre + queries[q] for q in qids], batch_size=BS, normalize_embeddings=True, convert_to_tensor=True)
        K = K_QUORA if d == "quora" else K_DEFAULT
        run, IX, SC, MU = {}, [], [], []
        for s0 in range(0, len(qids), 256):
            sc, ix = torch.topk((Q[s0:s0+256] @ D.T).float(), 1001, dim=1)
            for q, srow, irow in zip(qids[s0:s0+256], sc.cpu().numpy(), ix.cpu().numpy()):
                keep = [j for j, i in enumerate(irow) if doc_ids[i] != q][:1000]
                srow, irow = srow[keep], irow[keep]
                MU.append(float(srow.mean()))
                IX.append(irow[:K].astype(np.int32)); SC.append(srow[:K].astype(np.float32))
                run[q] = [(doc_ids[i], float(v)) for i, v in zip(irow[:K], srow[:K])]
        per = evaluate_run(run, qrels)
        pd.DataFrame(per_query_rows(per, d, mkey)).to_csv(os.path.join(OUT, f"perquery_{mkey}_{d}.csv"), index=False)
        np.savez_compressed(runf[:-4] + "_tmp.npz", qids=np.array(qids), idx=np.stack(IX), scores=np.stack(SC),
                            mu1000=np.array(MU, np.float32), k=K)
        os.replace(runf[:-4] + "_tmp.npz", runf)
        m = np.mean([v["ndcg@10"] for v in per.values()])
        print(f"{mkey:8s} {d:18s} nDCG@10={m:.4f}  ({time.time()-t:.0f}s)", flush=True)
        del D, Q
        torch.cuda.empty_cache()
