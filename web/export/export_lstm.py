"""Train the browser LSTM text classifier on SIX 20-Newsgroups topics and export
ONNX + vocab so the browser can tokenize text and classify it live.

Fixes vs. the first version: classify from a MASKED MEAN POOL over the BiLSTM
outputs (real tokens only) instead of the final hidden state hn[0], which sat on
a PAD position for post-padded sequences and badly degraded quality. Also adds
dropout and more epochs.

Output: web/public/models/lstm_text.onnx + lstm_vocab.json
Run on a GPU: python web/export/export_lstm.py
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

VOCAB, MAXLEN, PAD = 8000, 80, 0
CATS = ["rec.sport.hockey", "rec.autos", "sci.med", "sci.space", "comp.graphics", "talk.politics.mideast"]
LABELS = ["Hockey", "Cars", "Medicine", "Space", "Computer graphics", "Mideast politics"]
NC = len(CATS)

ng = fetch_20newsgroups(subset="all", categories=CATS, remove=("headers", "footers", "quotes"), random_state=42)
texts, labels = ng.data, ng.target

def tok(t): return re.findall(r"[a-z]+", t.lower())

# Build the vocabulary from the TRAINING split only (no leakage).
idx = np.random.permutation(len(texts)); sp = int(0.85 * len(texts))
tr, te = idx[:sp], idx[sp:]  # train / test indices
cnt = Counter()
for i in tr: cnt.update(tok(texts[i]))
words = [w for w, _ in cnt.most_common(VOCAB - 2)]
w2i = {w: i + 2 for i, w in enumerate(words)}  # 0=pad, 1=unk

def enc(t):
    ix = [w2i.get(x, 1) for x in tok(t)][:MAXLEN]
    return ix + [0] * (MAXLEN - len(ix))

X = np.array([enc(t) for t in texts], dtype=np.int64)
y = np.array(labels, dtype=np.int64)
Xt, yt = torch.tensor(X), torch.tensor(y)


class LSTMClf(nn.Module):
    def __init__(self, v=VOCAB, e=128, h=192, pad=PAD):
        super().__init__()
        self.pad = pad
        self.emb = nn.Embedding(v, e, padding_idx=pad)
        self.drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(e, h, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(h * 2, NC)

    def forward(self, x):
        mask = (x != self.pad).unsqueeze(-1).float()          # (B, T, 1) 1 on real tokens
        out, _ = self.lstm(self.drop(self.emb(x)))            # (B, T, 2h)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # mean over real tokens
        return self.fc(self.drop(pooled))


m = LSTMClf().to(DEV)
opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-5)
lossf = nn.CrossEntropyLoss()
bs, EPOCHS = 64, 18
for ep in range(EPOCHS):
    m.train(); perm = np.random.permutation(tr)
    for i in range(0, len(perm), bs):
        b = perm[i:i + bs]
        loss = lossf(m(Xt[b].to(DEV)), yt[b].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(Xt[te].to(DEV)).argmax(1).cpu() == yt[te]).float().mean().item()
    print(f"  epoch {ep+1}/{EPOCHS} test acc {acc:.3f}")

# Sanity check on the demo sentences — should each land on the right topic.
checks = [
    ("Hockey", "The goalie made an incredible save in overtime to win the playoff game."),
    ("Cars", "The new turbocharged engine gets great mileage but the transmission is rough."),
    ("Medicine", "The patient was prescribed antibiotics after the doctor diagnosed an infection."),
    ("Space", "The spacecraft entered orbit around the moon after a three-day journey."),
]
m.eval()
with torch.no_grad():
    print("\nSanity check:")
    for want, t in checks:
        p = torch.softmax(m(torch.tensor([enc(t)]).to(DEV))[0], 0)
        top = int(p.argmax())
        flag = "OK " if LABELS[top] == want else "XX "
        print(f"  {flag}want {want:8} got {LABELS[top]:18} ({p[top]*100:4.1f}%)")

m.eval().cpu()
torch.onnx.export(m, torch.zeros(1, MAXLEN, dtype=torch.long), os.path.join(OUT, "lstm_text.onnx"),
                  input_names=["tokens"], output_names=["logits"],
                  dynamic_axes={"tokens": {0: "b"}, "logits": {0: "b"}}, opset_version=13)
json.dump({"word2idx": w2i, "maxlen": MAXLEN, "labels": LABELS},
          open(os.path.join(OUT, "lstm_vocab.json"), "w"))
print("\nexported lstm_text.onnx + lstm_vocab.json  (", NC, "topics, vocab", len(w2i), ")")
