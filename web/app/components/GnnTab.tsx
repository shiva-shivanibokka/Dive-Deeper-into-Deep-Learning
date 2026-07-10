"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type Node = { x: number; y: number; pred: number; true: number };
type Data = { topics: string[]; testAcc: number; nodes: Node[]; adj: number[][] };

const COLORS = ["#22d3ee", "#a78bfa", "#f472b6", "#a3e635", "#fbbf24", "#fb7185", "#38bdf8"];
const SIZE = 520;

export default function GnnTab() {
  const [data, setData] = useState<Data | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => { fetch("/models/gnn_cora.json").then((r) => r.json()).then(setData); }, []);

  const P = (n: Node) => [n.x * (SIZE - 24) + 12, n.y * (SIZE - 24) + 12] as const;

  const draw = useCallback((d: Data, selected: number | null) => {
    const ctx = canvas.current!.getContext("2d")!;
    ctx.clearRect(0, 0, SIZE, SIZE);
    if (selected != null) {
      const [sx, sy] = P(d.nodes[selected]);
      ctx.strokeStyle = "rgba(167,139,250,0.55)"; ctx.lineWidth = 1;
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
      ctx.arc(x, y, isSel ? 7 : isNb ? 5 : 2.6, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[n.pred];
      ctx.fill();
      if (isSel) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5; ctx.stroke(); }
      else if (isNb) { ctx.strokeStyle = "rgba(255,255,255,0.5)"; ctx.lineWidth = 1; ctx.stroke(); }
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (data) draw(data, sel); }, [data, sel, draw]);

  const nearest = (e: React.MouseEvent) => {
    const r = canvas.current!.getBoundingClientRect();
    const cx = ((e.clientX - r.left) / r.width) * SIZE, cy = ((e.clientY - r.top) / r.height) * SIZE;
    let best = -1, bd = 1e9;
    data!.nodes.forEach((n, i) => {
      const [px, py] = P(n); const dist = (px - cx) ** 2 + (py - cy) ** 2;
      if (dist < bd) { bd = dist; best = i; }
    });
    return bd < 120 ? { i: best, ex: e.clientX - r.left, ey: e.clientY - r.top } : null;
  };

  if (!data) return <p className="note">loading graph…</p>;
  const n = sel != null ? data.nodes[sel] : null;

  return (
    <div className="demo">
      <div style={{ display: "grid", gridTemplateColumns: `${SIZE}px 1fr`, gap: "2rem", alignItems: "start" }}>
        <div style={{ position: "relative" }}
          onMouseMove={(e) => { const h = nearest(e); setHover(h ? { i: h.i, x: h.ex, y: h.ey } : null); }}
          onMouseLeave={() => setHover(null)}>
          <div className="canvas-frame"><canvas ref={canvas} width={SIZE} height={SIZE}
            onClick={() => setSel(hover ? hover.i : null)} style={{ width: SIZE, maxWidth: "100%", cursor: "pointer" }} /></div>
          {hover && (
            <div className="tip" style={{ position: "absolute", left: hover.x + 12, top: hover.y - 8, pointerEvents: "none" }}>
              <div className="pop" style={{ position: "static", opacity: 1, visibility: "visible", transform: "none" }}>
                Paper #{hover.i} — predicted <b style={{ color: COLORS[data.nodes[hover.i].pred] }}>{data.topics[data.nodes[hover.i].pred]}</b>
              </div>
            </div>
          )}
        </div>
        <div className="results">
          <div className="tile" style={{ textAlign: "left" }}>
            <div className="v" style={{ color: "var(--cyan)" }}>{(data.testAcc * 100).toFixed(0)}%</div>
            <div className="k">test accuracy from just 140 labels</div>
          </div>
          <p className="note">Each dot is a paper (t-SNE of the GCN&apos;s learned embeddings), colored by predicted topic. <strong>Hover</strong> to identify a paper; <strong>click</strong> to light up its citation links.</p>
          {n && (
            <div className="callout">
              <div>Paper #{sel} · <strong>{data.adj[sel!].length}</strong> citations</div>
              <div style={{ marginTop: ".3rem" }}>Predicted: <span style={{ color: COLORS[n.pred] }}>{data.topics[n.pred]}</span></div>
              <div>Actual: <span style={{ color: COLORS[n.true] }}>{data.topics[n.true]}</span> {n.pred === n.true ? "✓" : "✗"}</div>
            </div>
          )}
        </div>
      </div>
      {/* topic legend, in a row at the bottom */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem 1.4rem", marginTop: ".4rem", justifyContent: "center" }}>
        {data.topics.map((t, i) => (
          <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: ".45rem", fontSize: ".82rem", color: "var(--muted)" }}>
            <span style={{ width: 11, height: 11, borderRadius: "50%", background: COLORS[i] }} /> {t}
          </span>
        ))}
      </div>
    </div>
  );
}
