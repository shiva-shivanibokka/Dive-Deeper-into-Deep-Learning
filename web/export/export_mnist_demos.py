"""Train small MNIST demo models and export them to ONNX for the browser app.
Purpose-built tiny models (fast browser inference) — NOT the full notebook models.
Outputs land in web/public/models/.
"""
import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "public", "models"))
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print("device:", DEV, "| out:", OUT)

tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])  # [-1,1]
ds = torchvision.datasets.MNIST("data", train=True, download=True, transform=tf)
dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

# also a [0,1] loader for the classifier/autoencoder
tf01 = T.ToTensor()
ds01 = torchvision.datasets.MNIST("data", train=True, download=True, transform=tf01)
dl01 = DataLoader(ds01, batch_size=256, shuffle=True, num_workers=0)

LATENT = 8

# ---------------- VAE (export decoder: latent -> image) ----------------
class VAE(nn.Module):
    def __init__(self, z=LATENT):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, z); self.lv = nn.Linear(256, z)
        self.dec = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
    def forward(self, x):
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv

class VAEDecoder(nn.Module):
    def __init__(self, dec): super().__init__(); self.dec = dec
    def forward(self, z): return self.dec(z).view(-1, 1, 28, 28)

def train_vae(epochs=8):
    m = VAE().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for e in range(epochs):
        tot = 0
        for x, _ in dl:
            x = x.to(DEV); xr, mu, lv = m(x)
            rec = F.mse_loss(xr, x.view(x.size(0), -1), reduction="sum")
            kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
            loss = (rec + kld) / x.size(0)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  VAE epoch {e+1}/{epochs} loss {tot/len(dl):.2f}")
    return m

vae = train_vae()
dec = VAEDecoder(vae.dec).eval().cpu()
torch.onnx.export(dec, torch.randn(1, LATENT), os.path.join(OUT, "vae_decoder.onnx"),
                  input_names=["z"], output_names=["image"],
                  dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}}, opset_version=13)
print("exported vae_decoder.onnx  (latent dim", LATENT, ")")

# ---------------- DCGAN generator (noise -> image) ----------------
ZDIM = 32
class Gen(nn.Module):
    def __init__(self, z=ZDIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 784), nn.Tanh())
    def forward(self, z): return self.net(z).view(-1, 1, 28, 28)

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
                                 nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))
    def forward(self, x): return self.net(x)

def train_gan(epochs=12):
    g, d = Gen().to(DEV), Disc().to(DEV)
    og = torch.optim.Adam(g.parameters(), 2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(d.parameters(), 2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    for e in range(epochs):
        for x, _ in dl:
            x = x.to(DEV); b = x.size(0)
            z = torch.randn(b, ZDIM, device=DEV); fake = g(z)
            od.zero_grad()
            ld = bce(d(x), torch.ones(b, 1, device=DEV)) + bce(d(fake.detach()), torch.zeros(b, 1, device=DEV))
            ld.backward(); od.step()
            og.zero_grad()
            lg = bce(d(fake), torch.ones(b, 1, device=DEV)); lg.backward(); og.step()
        print(f"  GAN epoch {e+1}/{epochs}  D {ld.item():.2f}  G {lg.item():.2f}")
    return g

gen = train_gan().eval().cpu()
torch.onnx.export(gen, torch.randn(2, ZDIM), os.path.join(OUT, "gan_generator.onnx"),
                  input_names=["z"], output_names=["image"],
                  dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}}, opset_version=13)
print("exported gan_generator.onnx  (noise dim", ZDIM, ")")

# ---------------- CNN classifier (image -> logits) for draw-pad ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64); self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.flatten(1); x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_cnn(epochs=3):
    m = CNN().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for e in range(epochs):
        correct = tot = 0
        for x, y in dl01:
            x, y = x.to(DEV), y.to(DEV)
            out = m(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (out.argmax(1) == y).sum().item(); tot += y.size(0)
        print(f"  CNN epoch {e+1}/{epochs}  train acc {correct/tot:.3f}")
    return m

cnn = train_cnn().eval().cpu()
torch.onnx.export(cnn, torch.randn(1, 1, 28, 28), os.path.join(OUT, "mnist_cnn.onnx"),
                  input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=13)
print("exported mnist_cnn.onnx")

# ---------------- Autoencoder (image -> reconstruction) ----------------
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 16), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())
    def forward(self, x): return self.dec(self.enc(x)).view(-1, 1, 28, 28)

def train_ae(epochs=6):
    m = AE().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for e in range(epochs):
        tot = 0
        for x, _ in dl01:
            x = x.to(DEV); xr = m(x); loss = F.mse_loss(xr, x)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  AE epoch {e+1}/{epochs}  mse {tot/len(dl01):.4f}")
    return m

ae = train_ae().eval().cpu()
torch.onnx.export(ae, torch.randn(1, 1, 28, 28), os.path.join(OUT, "autoencoder.onnx"),
                  input_names=["image"], output_names=["recon"],
                  dynamic_axes={"image": {0: "batch"}, "recon": {0: "batch"}}, opset_version=13)
print("exported autoencoder.onnx")

print("\nALL MNIST DEMO MODELS EXPORTED to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f, os.path.getsize(os.path.join(OUT, f)), "bytes")
