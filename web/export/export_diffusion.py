"""Train a DDPM on MNIST and export the denoising U-Net to ONNX plus the noise
schedule as JSON, so the browser can run the reverse sampling loop live.

Bigger than the first version: wider U-Net (base=48) with residual blocks, more
diffusion steps and more epochs, for noticeably cleaner noise-to-digit samples.
The browser reads the step count from diffusion_schedule.json, so no frontend
change is needed. Larger ONNX = a slightly longer first load + sampling loop.

Output: web/public/models/diffusion_unet.onnx + diffusion_schedule.json
Run on a GPU: python web/export/export_diffusion.py
"""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "public", "models"))
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print("device", DEV)

STEPS = 60  # more steps than before -> smoother denoising (browser runs one U-Net call per step)
betas = torch.linspace(1e-4, 0.02, STEPS)
alphas = 1 - betas
abars = torch.cumprod(alphas, 0)

tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
ds = torchvision.datasets.MNIST("data", train=True, download=True, transform=tf)
dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

def temb(t, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)

class Block(nn.Module):
    """Two conv layers + timestep bias, with a residual shortcut."""
    def __init__(self, i, o, td):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, padding=1); self.c2 = nn.Conv2d(o, o, 3, padding=1)
        self.t = nn.Linear(td, o); self.n1 = nn.GroupNorm(8, o); self.n2 = nn.GroupNorm(8, o)
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
    def forward(self, x, t):
        h = F.silu(self.n1(self.c1(x))); h = h + self.t(t)[:, :, None, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, base=48, td=128):
        super().__init__()
        self.td = td
        self.mlp = nn.Sequential(nn.Linear(td, td), nn.SiLU(), nn.Linear(td, td))
        self.d1 = Block(1, base, td); self.d2 = Block(base, base * 2, td)
        self.pool = nn.MaxPool2d(2)
        self.mid1 = Block(base * 2, base * 2, td); self.mid2 = Block(base * 2, base * 2, td)
        self.u2 = Block(base * 4, base, td); self.u1 = Block(base * 2, base, td)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x, t):
        te = self.mlp(temb(t, self.td))
        d1 = self.d1(x, te); d2 = self.d2(self.pool(d1), te)
        m = self.mid2(self.mid1(self.pool(d2), te), te)
        u2 = self.u2(torch.cat([F.interpolate(m, scale_factor=2, mode="nearest"), d2], 1), te)
        u1 = self.u1(torch.cat([F.interpolate(u2, scale_factor=2, mode="nearest"), d1], 1), te)
        return self.out(u1)

m = UNet().to(DEV)
opt = torch.optim.Adam(m.parameters(), 2e-4)
print("params", sum(p.numel() for p in m.parameters()))
EPOCHS = 30
for e in range(EPOCHS):
    tot = 0
    for x, _ in dl:
        x = x.to(DEV); t = torch.randint(0, STEPS, (x.size(0),), device=DEV)
        noise = torch.randn_like(x)
        ab = abars.to(DEV)[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise
        loss = F.mse_loss(m(xt, t), noise)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"  epoch {e+1}/{EPOCHS} loss {tot/len(dl):.4f}")

m.eval().cpu()
torch.onnx.export(m, (torch.randn(1, 1, 28, 28), torch.zeros(1, dtype=torch.long)),
                  os.path.join(OUT, "diffusion_unet.onnx"),
                  input_names=["x", "t"], output_names=["noise"],
                  dynamic_axes={"x": {0: "b"}, "t": {0: "b"}, "noise": {0: "b"}}, opset_version=13)
json.dump({"steps": STEPS, "betas": betas.tolist(), "alphas": alphas.tolist(), "abars": abars.tolist()},
          open(os.path.join(OUT, "diffusion_schedule.json"), "w"))
print("exported diffusion_unet.onnx + diffusion_schedule.json")
