"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";

type Sample = { label: string; pixels: number[] };
type Data = { classes: string[]; samples: Sample[] };

export default function VitTab() {
  const [data, setData] = useState<Data | null>(null);
  const [sel, setSel] = useState(0);
  const [probs, setProbs] = useState<number[] | null>(null);
  const bigCanvas = useRef<HTMLCanvasElement>(null);
  const thumbs = useRef<(HTMLCanvasElement | null)[]>([]);

  useEffect(() => {
    fetch("/models/vit_samples.json").then((r) => r.json()).then(setData);
  }, []);

  useEffect(() => {
    if (!data) return;
    data.samples.forEach((s, i) => {
      const c = thumbs.current[i];
      if (c) renderGray(c, Float32Array.from(s.pixels), 28, 28);
    });
  }, [data]);

  const classify = useCallback(async (d: Data, i: number) => {
    const px = Float32Array.from(d.samples[i].pixels);
    renderGray(bigCanvas.current!, px, 28, 28);
    const sess = await getSession("vit.onnx");
    const out = await sess.run({ image: new ort.Tensor("float32", px, [1, 1, 28, 28]) });
    setProbs(softmax(Array.from(out.logits.data as Float32Array)));
  }, []);

  useEffect(() => { if (data) classify(data, sel); }, [data, sel, classify]);

  if (!data) return <p style={{ color: "var(--muted)" }}>loading model…</p>;
  const order = probs ? probs.map((p, i) => [p, i] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, 3) : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <p style={{ color: "var(--muted)", fontSize: ".8rem", margin: 0 }}>Pick a Fashion-MNIST image — the Vision Transformer classifies it by attending over its patches.</p>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {data.samples.map((s, i) => (
          <canvas key={i} ref={(el) => { thumbs.current[i] = el; }} width={48} height={48}
            onClick={() => setSel(i)} title={s.label}
            style={{ width: 48, height: 48, borderRadius: 6, cursor: "pointer", background: "#000", imageRendering: "pixelated",
              border: `2px solid ${i === sel ? "var(--accent)" : "var(--border)"}` }} />
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "1.5rem", alignItems: "center" }}>
        <canvas ref={bigCanvas} width={140} height={140}
          style={{ width: 140, height: 140, borderRadius: 8, border: "1px solid var(--border)", background: "#000", imageRendering: "pixelated" }} />
        <div>
          <div style={{ color: "var(--muted)", fontSize: ".78rem", marginBottom: ".3rem" }}>True label: {data.samples[sel].label}</div>
          {order.map(([p, i]) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: ".5rem", fontSize: ".82rem", marginBottom: 3 }}>
              <span style={{ width: 80, color: "var(--muted)" }}>{data.classes[i]}</span>
              <div style={{ flex: 1, maxWidth: 220, background: "var(--panel-2)", borderRadius: 4, height: 12 }}>
                <div style={{ width: `${p * 100}%`, height: "100%", background: p === order[0][0] ? "var(--accent)" : "var(--accent-2)", borderRadius: 4 }} />
              </div>
              <span style={{ width: 44, textAlign: "right", color: "var(--muted)" }}>{(p * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function softmax(a: number[]) {
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
}
