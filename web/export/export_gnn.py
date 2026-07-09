"""Train the GCN on Cora, then precompute everything the browser needs for an interactive
graph explorer: t-SNE node positions, predicted+true class per node, and adjacency.
(A full GCN forward pass needs the whole graph, so we ship its outputs, not the model.)
Output: web/public/models/gnn_cora.json
"""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "public", "models"))
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)

TOPICS = ["Case-Based", "Genetic Algorithms", "Neural Networks", "Probabilistic Methods",
          "Reinforcement Learning", "Rule Learning", "Theory"]
data = Planetoid(root="data/Cora", name="Cora")[0]

class GCN(nn.Module):
    def __init__(self, i, h, o):
        super().__init__(); self.c1 = GCNConv(i, h); self.c2 = GCNConv(h, o)
    def forward(self, x, ei, emb=False):
        h = F.relu(self.c1(x, ei)); hd = F.dropout(h, 0.5, self.training)
        out = self.c2(hd, ei)
        return (out, h) if emb else out

m = GCN(data.num_node_features, 64, 7)
opt = torch.optim.Adam(m.parameters(), 0.01, weight_decay=5e-4)
for e in range(200):
    m.train(); opt.zero_grad()
    out = m(data.x, data.edge_index)
    F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward(); opt.step()
m.eval()
with torch.no_grad():
    logits, emb = m(data.x, data.edge_index, emb=True)
    pred = logits.argmax(1)
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
print(f"GCN test acc {test_acc:.3f}")

print("running t-SNE on node embeddings...")
pos = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(emb.numpy())
pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0))  # normalize to [0,1]

# adjacency list
ei = data.edge_index.numpy()
adj = [[] for _ in range(data.num_nodes)]
for s, d in zip(ei[0], ei[1]):
    if d not in adj[s]:
        adj[s].append(int(d))

nodes = [{"x": round(float(pos[i, 0]), 4), "y": round(float(pos[i, 1]), 4),
          "pred": int(pred[i]), "true": int(data.y[i])} for i in range(data.num_nodes)]
json.dump({"topics": TOPICS, "testAcc": round(test_acc, 3), "nodes": nodes, "adj": adj},
          open(os.path.join(OUT, "gnn_cora.json"), "w"))
print("exported gnn_cora.json  (", data.num_nodes, "nodes )")
