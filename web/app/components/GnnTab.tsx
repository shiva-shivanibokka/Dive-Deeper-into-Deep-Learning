"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type Node = { x: number; y: number; pred: number; true: number };
type Data = { topics: string[]; testAcc: number; nodes: Node[]; adj: number[][] };

const COLORS = ["#22d3ee", "#a78bfa", "#f472b6", "#a3e635", "#fbbf24", "#fb7185", "#38bdf8"];
const SIZE = 440;

export default function GnnTab() {
  const [data, setData] = useState<Data | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => { fetch("/models/gnn_cora.json").then((r) => r.json()).then(setData); }, []);

  const draw = useCallback((d: Data, selected: number | null) => {
    const ctx = canvas.current!.getContext("2d")!;
    ctx.clearRect(0, 0, SIZE, SIZE);
    const P = (n: Node) => [n.x * (SIZE - 20) + 10, n.y * (SIZE - 20) + 10] as const;
    if (selected != null) {
      const [sx, sy] = P(d.nodes[selected]);
      ctx.strokeStyle = "rgba(167,139,250,0.55)";
      ctx.lineWidth = 1;
      for (const nb of d.adj[selected]) {
        const [nx, ny] = P(d.nodes[nb]);
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(nx, ny); ctx.stroke();
      }
    }
    d.nodes.forEach((n, i) => {
      const [x, y] = P(n);
      const isSel = i === selected;
      const isNb = selected != null && d.adj[selected].includes(i);
      ctx.beginPath();
      ctx.arc(x, y, isSel ? 6 : isNb ? 4 : 2.2, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[n.pred];
      ctx.globalAlpha = selected == null || isSel || isNb ? 1 : 0.3;
      ctx.fill();
      if (isSel) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); }
    });
    ctx.globalAlpha = 1;
  }, []);

  useEffect(() => { if (data) draw(data, sel); }, [data, sel, draw]);

  const click = (e: React.MouseEvent) => {
    if (!data) return;
    const r = canvas.current!.getBoundingClientRect();
    const cx = ((e.clientX - r.left) / r.width) * SIZE, cy = ((e.clientY - r.top) / r.height) * SIZE;
    let best = -1, bd = 1e9;
    data.nodes.forEach((n, i) => {
      const dx = n.x * (SIZE - 20) + 10 - cx, dy = n.y * (SIZE - 20) + 10 - cy, dist = dx * dx + dy * dy;
      if (dist < bd) { bd = dist; best = i; }
    });
    if (bd < 200) setSel(best);
  };

  if (!data) return <p className="note">loading graph…</p>;
  const n = sel != null ? data.nodes[sel] : null;

  return (
    <div className="demo" style={{ gridTemplateColumns: `${SIZE}px 1fr`, display: "grid", alignItems: "start", gap: "2rem" }}>
      <div className="canvas-frame"><canvas ref={canvas} width={SIZE} height={SIZE} onClick={click} style={{ width: SIZE, maxWidth: "100%", cursor: "pointer" }} /></div>
      <div className="results">
        <div className="tile" style={{ textAlign: "left" }}>
          <div className="v" style={{ color: "var(--cyan)" }}>{(data.testAcc * 100).toFixed(0)}%</div>
          <div className="k">test accuracy from just 140 labels</div>
        </div>
        <p className="note">Each dot is a paper (t-SNE of the GCN&apos;s learned embeddings), colored by predicted topic. <strong>Click a node</strong> to see its citation links and prediction.</p>
        {n && (
          <div className="callout">
            <div>Paper #{sel} · <strong>{data.adj[sel!].length}</strong> citations</div>
            <div style={{ marginTop: ".3rem" }}>Predicted: <span style={{ color: COLORS[n.pred] }}>{data.topics[n.pred]}</span></div>
            <div>Actual: <span style={{ color: COLORS[n.true] }}>{data.topics[n.true]}</span> {n.pred === n.true ? "✓" : "✗"}</div>
          </div>
        )}
        <div style={{ display: "grid", gap: 4 }}>
          {data.topics.map((t, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: ".4rem", fontSize: ".78rem", color: "var(--muted)" }}>
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: COLORS[i] }} /> {t}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
