# Dive Deeper into Deep Learning

A hands-on, from-scratch tour of modern deep learning in **PyTorch** — eight self-contained notebooks that build every major architecture family from first principles, each with plain-language explanations and a "how to read this chart" guide for every visualization.

Every model is also **interactive in a companion web app** that runs entirely in your browser — no install, no server.

### ▶ [**Live demo → dive-deeper-deep-learning-shiv-a.vercel.app**](https://dive-deeper-deep-learning-shiv-a.vercel.app)

---

## The Notebooks

| # | Notebook | Architectures | Dataset |
|---|----------|---------------|---------|
| 01 | [Feedforward Networks](01_feedforward_networks.ipynb) | MLP, weight init, BatchNorm/Dropout, Autoencoder, anomaly detection | Adult Income, MNIST |
| 02 | [Convolutional Networks](02_convolutional_networks.ipynb) | CNN from scratch, Grad-CAM, transfer learning (ResNet-18) | CIFAR-10 |
| 03 | [Sequence Models](03_sequence_models.ipynb) | RNN, LSTM, GRU, BiLSTM, attention, gradient clipping | 20 Newsgroups |
| 04 | [Attention & Transformers](04_attention_transformers.ipynb) | Scaled dot-product attention, multi-head attention, positional encoding, DistilBERT fine-tuning | 20 Newsgroups |
| 05 | [Generative Models](05_generative_models.ipynb) | VAE, DCGAN, WGAN, conditional VAE, FID | MNIST |
| 06 | [Diffusion Models](06_diffusion_models.ipynb) | DDPM from scratch, noise schedule, denoising U-Net | MNIST |
| 07 | [Vision Transformer](07_vision_transformer.ipynb) | ViT from scratch, patch embedding, attention maps | Fashion-MNIST |
| 08 | [Graph Neural Networks](08_graph_neural_networks.ipynb) | GCN, message passing, node classification | Cora |

They're ordered by **data shape** — flat vectors → grids → sequences → generation → graphs — so each notebook builds on the last.

---

## What Makes These Different

- **From scratch, then production.** Notebook 02 builds a CNN by hand *and* fine-tunes a pretrained ResNet. Notebook 04 implements attention from the equations *and* fine-tunes DistilBERT. You see both the mechanism and the real-world tool.
- **Every chart is explained.** No unlabeled plots. Each visualization has a dedicated "How to Read This Chart" section.
- **Cross-notebook throughlines.** Grad-CAM (nb 02) and ViT attention maps (nb 07) answer the same "where is the model looking?" question two ways. VAEs/GANs (nb 05) and Diffusion (nb 06) are compared head-to-head. The GNN (nb 08) is benchmarked against the MLP (nb 01).

---

## Quickstart

```bash
# 1. (Recommended) install a CUDA-matched PyTorch first if you have a GPU:
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2. Install everything else:
pip install -r requirements.txt
# 3. Launch:
jupyter notebook
```

Datasets download automatically on first run (into `data/`, git-ignored). A GPU is recommended for notebooks 02, 05, 06, and 07 but not required.

---

## Interactive Web Demo

**Live:** https://dive-deeper-deep-learning-shiv-a.vercel.app

The `web/` directory is a **Next.js** app that runs each trained model **client-side** via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) and [Transformers.js](https://huggingface.co/docs/transformers.js) — draw a digit, move latent-space sliders, watch diffusion denoise, classify your own text, all in the browser with zero backend. Deployed on **Vercel**.

```bash
cd web
npm install
npm run dev
```

---

## Repository Structure

```
.
├── 01_feedforward_networks.ipynb   ... 08_graph_neural_networks.ipynb
├── requirements.txt                # Python dependencies
├── web/                            # Next.js interactive demo (deployed to Vercel)
│   └── public/models/              # exported .onnx weights
└── data/                           # datasets (auto-downloaded, git-ignored)
```

---

## Tech Stack

**Modeling:** PyTorch · torchvision · PyTorch Geometric · Hugging Face Transformers · scikit-learn · XGBoost
**Serving:** ONNX · ONNX Runtime Web · Transformers.js · Next.js · Vercel

---

*Author: Shivani Bokka*
