"""Train nb03's LSTM text classifier (20 Newsgroups: hockey vs medicine) and export
ONNX + vocab so the browser can tokenize text and classify it live.
Output: web/public/models/lstm_text.onnx + lstm_vocab.json
"""
import os, json, re, numpy as np, torch, torch.nn as nn
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "public", "models"))
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB, MAXLEN = 5000, 60
cats = ["rec.sport.hockey", "sci.med"]
ng = fetch_20newsgroups(subset="all", categories=cats, remove=("headers", "footers", "quotes"), random_state=42)
texts, labels = ng.data, ng.target  # 0=hockey, 1=sci.med

def tok(t): return re.findall(r"[a-z]+", t.lower())
cnt = Counter()
for t in texts: cnt.update(tok(t))
words = [w for w, _ in cnt.most_common(VOCAB - 2)]
w2i = {w: i + 2 for i, w in enumerate(words)}  # 0=pad, 1=unk

def enc(t):
    ix = [w2i.get(x, 1) for x in tok(t)][:MAXLEN]
    return ix + [0] * (MAXLEN - len(ix))

X = np.array([enc(t) for t in texts], dtype=np.int64)
y = np.array(labels, dtype=np.int64)
idx = np.random.permutation(len(X)); sp = int(0.85 * len(X))
tr, te = idx[:sp], idx[sp:]
Xt, yt = torch.tensor(X), torch.tensor(y)

class LSTMClf(nn.Module):
    def __init__(self, v=VOCAB, e=128, h=128):
        super().__init__()
        self.emb = nn.Embedding(v, e, padding_idx=0)
        self.lstm = nn.LSTM(e, h, batch_first=True)
        self.fc = nn.Linear(h, 2)
    def forward(self, x):
        _, (hn, _) = self.lstm(self.emb(x))
        return self.fc(hn[-1])

m = LSTMClf().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
lossf = nn.CrossEntropyLoss()
bs = 64
for ep in range(6):
    m.train(); perm = np.random.permutation(tr)
    for i in range(0, len(perm), bs):
        b = perm[i:i + bs]
        xb, yb = Xt[b].to(DEV), yt[b].to(DEV)
        loss = lossf(m(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(Xt[te].to(DEV)).argmax(1).cpu() == yt[te]).float().mean().item()
    print(f"  epoch {ep+1}/6 test acc {acc:.3f}")

m.eval().cpu()
torch.onnx.export(m, torch.zeros(1, MAXLEN, dtype=torch.long), os.path.join(OUT, "lstm_text.onnx"),
                  input_names=["tokens"], output_names=["logits"],
                  dynamic_axes={"tokens": {0: "b"}, "logits": {0: "b"}}, opset_version=13)
json.dump({"word2idx": w2i, "maxlen": MAXLEN, "labels": ["Hockey / sports", "Medicine / health"]},
          open(os.path.join(OUT, "lstm_vocab.json"), "w"))
print("exported lstm_text.onnx + lstm_vocab.json  (vocab", len(w2i), ")")
