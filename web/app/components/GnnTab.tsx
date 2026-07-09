"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type Node = { x: number; y: number; pred: number; true: number };
type Data = { topics: string[]; testAcc: number; nodes: Node[]; adj: number[][] };

const COLORS = ["#60a5fa", "#6ee7b7", "#f59e0b", "#f472b6", "#a78bfa", "#f87171", "#34d399"];
const SIZE = 460;

export default function GnnTab() {
  const [data, setData] = useState<Data | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetch("/models/gnn_cora.json").then((r) => r.json()).then(setData);
  }, []);

  const draw = useCallback((d: Data, selected: number | null) => {
    const ctx = canvas.current!.getContext("2d")!;
    ctx.clearRect(0, 0, SIZE, SIZE);
    const P = (n: Node) => [n.x * (SIZE - 20) + 10, n.y * (SIZE - 20) + 10] as const;
    // edges of the selected node
    if (selected != null) {
      const [sx, sy] = P(d.nodes[selected]);
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      for (const nb of d.adj[selected]) {
        const [nx, ny] = P(d.nodes[nb]);
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(nx, ny); ctx.stroke();
      }
    }
    // nodes
    d.nodes.forEach((n, i) => {
      const [x, y] = P(n);
      const isSel = i === selected;
      const isNb = selected != null && d.adj[selected].includes(i);
      ctx.beginPath();
      ctx.arc(x, y, isSel ? 6 : isNb ? 4 : 2.2, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[n.pred];
      ctx.globalAlpha = selected == null || isSel || isNb ? 1 : 0.35;
      ctx.fill();
      if (isSel) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); }
    });
    ctx.globalAlpha = 1;
  }, []);

  useEffect(() => { if (data) draw(data, sel); }, [data, sel, draw]);

  const click = (e: React.MouseEvent) => {
    if (!data) return;
    const r = canvas.current!.getBoundingClientRect();
    const cx = ((e.clientX - r.left) / r.width) * SIZE;
    const cy = ((e.clientY - r.top) / r.height) * SIZE;
    let best = -1, bd = 1e9;
    data.nodes.forEach((n, i) => {
      const dx = n.x * (SIZE - 20) + 10 - cx, dy = n.y * (SIZE - 20) + 10 - cy;
      const dist = dx * dx + dy * dy;
      if (dist < bd) { bd = dist; best = i; }
    });
    if (bd < 200) setSel(best);
  };

  if (!data) return <p style={{ color: "var(--muted)" }}>loading graph…</p>;
  const n = sel != null ? data.nodes[sel] : null;

  return (
    <div style={{ display: "grid", gridTemplateColumns: `${SIZE}px 1fr`, gap: "1.5rem", alignItems: "start" }}>
      <canvas ref={canvas} width={SIZE} height={SIZE} onClick={click}
        style={{ width: SIZE, height: SIZE, maxWidth: "100%", borderRadius: 10, border: "1px solid var(--border)", background: "var(--panel-2)", cursor: "pointer" }} />
      <div>
        <p style={{ color: "var(--muted)", fontSize: ".8rem", marginTop: 0 }}>
          Each dot is a paper (t-SNE of the GCN&apos;s learned embeddings), colored by predicted topic.
          Test accuracy <strong style={{ color: "var(--accent)" }}>{(data.testAcc * 100).toFixed(0)}%</strong> from just 140 labels. Click a node to see its citations.
        </p>
        {n && (
          <div style={{ margin: ".8rem 0", padding: ".8rem", background: "var(--panel-2)", borderRadius: 8, fontSize: ".85rem" }}>
            <div>Paper #{sel} · <strong>{data.adj[sel!].length}</strong> citations</div>
            <div style={{ marginTop: ".3rem" }}>Predicted: <span style={{ color: COLORS[n.pred] }}>{data.topics[n.pred]}</span></div>
            <div>Actual: <span style={{ color: COLORS[n.true] }}>{data.topics[n.true]}</span> {n.pred === n.true ? "✓" : "✗"}</div>
          </div>
        )}
        <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: ".5rem" }}>
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
