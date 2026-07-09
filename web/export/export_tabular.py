"""Train a small MLP on curated Adult-income features and export ONNX + form metadata.
Interpretable feature set so the browser form is clean. Output: web/public/models/tabular_mlp.onnx
and web/public/models/tabular_meta.json (feature order + normalization constants for the JS form).
"""
import os, json, numpy as np, torch, torch.nn as nn
from sklearn.datasets import fetch_openml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "public", "models"))
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)

print("loading Adult (OpenML)...")
adult = fetch_openml("adult", version=2, as_frame=True, data_home="data/openml")
df = adult.frame.dropna()

# Curated, interpretable features
NUM = ["age", "education-num", "hours-per-week", "capital-gain"]
X_num = df[NUM].astype(float).values
sex = (df["sex"].astype(str).values == "Male").astype(np.float32)          # 1=Male
married = df["marital-status"].astype(str).str.startswith("Married").values.astype(np.float32)
y = (df["class"].astype(str).values == ">50K").astype(np.float32)

# Standardize numerics; keep constants for the JS form
mean = X_num.mean(0); std = X_num.std(0)
X_num_s = (X_num - mean) / std
X = np.concatenate([X_num_s, sex[:, None], married[:, None]], axis=1).astype(np.float32)
FEATURES = NUM + ["sex_is_male", "is_married"]
print("features:", FEATURES, "| samples:", len(X), "| positive rate:", y.mean().round(3))

Xt = torch.tensor(X); yt = torch.tensor(y)[:, None]

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)  # logit

m = MLP(X.shape[1]); opt = torch.optim.Adam(m.parameters(), 1e-3)
lossf = nn.BCEWithLogitsLoss()
n = len(X); idx = np.arange(n); split = int(0.85 * n)
np.random.shuffle(idx); tr, te = idx[:split], idx[split:]
for e in range(60):
    m.train(); opt.zero_grad()
    loss = lossf(m(Xt[tr]), yt[tr]); loss.backward(); opt.step()
    if (e + 1) % 20 == 0:
        m.eval()
        with torch.no_grad():
            acc = ((torch.sigmoid(m(Xt[te])) > 0.5).float() == yt[te]).float().mean().item()
        print(f"  epoch {e+1}  loss {loss.item():.3f}  test acc {acc:.3f}")

m.eval()
torch.onnx.export(m, torch.randn(1, X.shape[1]), os.path.join(OUT, "tabular_mlp.onnx"),
                  input_names=["features"], output_names=["logit"],
                  dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}}, opset_version=13)

meta = {
    "features": FEATURES,
    "numeric": NUM,
    "mean": mean.tolist(), "std": std.tolist(),
    "ranges": {"age": [17, 90], "education-num": [1, 16], "hours-per-week": [1, 99], "capital-gain": [0, 99999]},
    "defaults": {"age": 39, "education-num": 10, "hours-per-week": 40, "capital-gain": 0,
                 "sex_is_male": 1, "is_married": 1},
}
json.dump(meta, open(os.path.join(OUT, "tabular_meta.json"), "w"), indent=2)
print("exported tabular_mlp.onnx + tabular_meta.json")
