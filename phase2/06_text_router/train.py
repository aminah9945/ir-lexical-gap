"""E5 text router: settings fixed in PREREGISTRATION.md (Drive createdTime 2026-11-03T18:13:32Z). Resumable:
each (model, held-out dataset) writes preds/<model>__<ds>.csv and is skipped if it exists."""
import os, sys, json, random, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__)); A3 = os.path.join(HERE, "..", "03_experiments", "results")
QDIR = "/home/user/beir_dl"; os.makedirs(os.path.join(HERE, "preds"), exist_ok=True)
DS = ["nfcorpus", "scifact", "fiqa", "scidocs", "arguana", "trec-covid", "quora", "webis-touche2020"]
ENC = "sentence-transformers/all-MiniLM-L6-v2"; LR, EPOCHS, BS, MAXLEN, SEED, QUORA_CAP = 3e-5, 2, 32, 128, 42, 1500

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.enc = AutoModel.from_pretrained(ENC); self.head = torch.nn.Linear(self.enc.config.hidden_size, 2)
    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float(); return self.head((h * m).sum(1) / m.sum(1).clamp(min=1e-9))

def load(model):
    d = pd.read_csv(os.path.join(A3, "w1_router_perquery", f"A_{model}_std_k8.csv.gz"), dtype={"qid": str})[["dataset", "qid", "B", "D", "H"]]
    txt = {}
    for ds in DS:
        for l in open(os.path.join(QDIR, f"q_{ds}", "queries.jsonl")):
            j = json.loads(l); txt[(ds, j["_id"])] = j["text"]
    d["text"] = [txt[(a, b)] for a, b in zip(d.dataset, d.qid)]
    d["dB"] = d.B - d.H; d["dD"] = d.D - d.H
    return d

tok = AutoTokenizer.from_pretrained(ENC)
def batches(texts, bs):
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt"); yield i, e

for model in sys.argv[1:] or ["bgebase", "bge", "minilm"]:
    d = load(model)
    for test in DS:
        outp = os.path.join(HERE, "preds", f"{model}__{test}.csv")
        if os.path.exists(outp): continue
        t0 = time.time(); seed_all(SEED)
        tr = d[d.dataset != test]
        q = tr[tr.dataset == "quora"]
        if len(q) > QUORA_CAP: tr = pd.concat([tr[tr.dataset != "quora"], q.sample(QUORA_CAP, random_state=SEED)])
        tr = tr.sample(frac=1, random_state=SEED).reset_index(drop=True)
        net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=LR)
        Y = torch.tensor(tr[["dB", "dD"]].values, dtype=torch.float32); texts = tr.text.tolist()
        net.train()
        for ep in range(EPOCHS):
            perm = np.random.permutation(len(texts)); tx = [texts[i] for i in perm]; yy = Y[perm]
            for i, e in batches(tx, BS):
                loss = torch.nn.functional.mse_loss(net(e["input_ids"], e["attention_mask"]), yy[i:i + BS])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); te = d[d.dataset == test].reset_index(drop=True); P = []
        with torch.no_grad():
            for i, e in batches(te.text.tolist(), 128): P.append(net(e["input_ids"], e["attention_mask"]).numpy())
        P = np.concatenate(P); te["pB"] = P[:, 0]; te["pD"] = P[:, 1]
        te.drop(columns=["text"]).to_csv(outp, index=False)
        print(model, test, len(tr), f"{time.time() - t0:.0f}s", flush=True)
print("ALLDONE", flush=True)
