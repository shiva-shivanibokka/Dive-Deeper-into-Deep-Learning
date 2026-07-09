"""Train a small ViT (manual attention -> clean ONNX) on Fashion-MNIST and export the
classifier plus a handful of test images for the browser picker.
Output: web/public/models/vit.onnx + vit_samples.json
"""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "public", "models"))
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
tf = T.ToTensor()
train = torchvision.datasets.FashionMNIST("data", train=True, download=True, transform=tf)
test = torchvision.datasets.FashionMNIST("data", train=False, download=True, transform=tf)
dl = DataLoader(train, batch_size=256, shuffle=True, num_workers=0)

PATCH, DIM, HEADS, DEPTH = 7, 64, 4, 3
NP = (28 // PATCH) ** 2

class MHA(nn.Module):
    def __init__(self, d=DIM, h=HEADS):
        super().__init__(); self.h = h; self.hd = d // h
        self.qkv = nn.Linear(d, d * 3); self.proj = nn.Linear(d, d)
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.hd ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d=DIM):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = MHA(d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
    def forward(self, x):
        x = x + self.attn(self.n1(x)); return x + self.mlp(self.n2(x))

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(1, DIM, PATCH, PATCH)
        self.cls = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos = nn.Parameter(torch.zeros(1, NP + 1, DIM))
        nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM); self.head = nn.Linear(DIM, 10)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x[:, 0]))

m = ViT().to(DEV); opt = torch.optim.Adam(m.parameters(), 3e-4)
for ep in range(12):
    m.train()
    for x, y in dl:
        x, y = x.to(DEV), y.to(DEV)
        loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        xt = torch.stack([test[i][0] for i in range(1000)]).to(DEV)
        yt = torch.tensor([test[i][1] for i in range(1000)])
        acc = (m(xt).argmax(1).cpu() == yt).float().mean().item()
    print(f"  epoch {ep+1}/12 test acc(1k) {acc:.3f}")

m.eval().cpu()
torch.onnx.export(m, torch.randn(1, 1, 28, 28), os.path.join(OUT, "vit.onnx"),
                  input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "b"}, "logits": {0: "b"}}, opset_version=14)

# 12 varied test images for the browser picker
samples = []
seen = set()
for i in range(len(test)):
    img, lab = test[i]
    if lab in seen and len(seen) < 10:  # get one of each class first
        continue
    seen.add(lab)
    samples.append({"label": CLASSES[lab], "pixels": [round(float(p), 3) for p in img.flatten().tolist()]})
    if len(samples) >= 12:
        break
json.dump({"classes": CLASSES, "samples": samples}, open(os.path.join(OUT, "vit_samples.json"), "w"))
print("exported vit.onnx + vit_samples.json")
