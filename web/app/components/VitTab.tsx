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

  useEffect(() => { fetch("/models/vit_samples.json").then((r) => r.json()).then(setData); }, []);

  useEffect(() => {
    if (!data) return;
    data.samples.forEach((s, i) => { const c = thumbs.current[i]; if (c) renderGray(c, Float32Array.from(s.pixels), 28, 28); });
  }, [data]);

  const classify = useCallback(async (d: Data, i: number) => {
    const px = Float32Array.from(d.samples[i].pixels);
    renderGray(bigCanvas.current!, px, 28, 28);
    const sess = await getSession("vit.onnx");
    const out = await sess.run({ image: new ort.Tensor("float32", px, [1, 1, 28, 28]) });
    setProbs(softmax(Array.from(out.logits.data as Float32Array)));
  }, []);

  useEffect(() => { if (data) classify(data, sel); }, [data, sel, classify]);

  if (!data) return <p className="note">loading model…</p>;
  const order = probs ? probs.map((p, i) => [p, i] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, 3) : [];

  return (
    <div className="demo">
      <p className="section-label">Pick a Fashion-MNIST image — the ViT splits it into patches and classifies it</p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {data.samples.map((s, i) => (
          <canvas key={i} ref={(el) => { thumbs.current[i] = el; }} width={52} height={52} onClick={() => setSel(i)} title={s.label}
            style={{ width: 52, height: 52, borderRadius: 8, cursor: "pointer", background: "#000", imageRendering: "pixelated",
              border: `2px solid ${i === sel ? "var(--cyan)" : "var(--border)"}`, boxShadow: i === sel ? "0 0 14px -2px var(--cyan)" : "none" }} />
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "2rem", alignItems: "center" }}>
        <div style={{ textAlign: "center" }}>
          <div className="canvas-frame"><canvas ref={bigCanvas} width={150} height={150} style={{ width: 150, height: 150 }} /></div>
          <div className="section-label" style={{ justifyContent: "center", margin: ".45rem 0 0" }}>True: {data.samples[sel].label}</div>
        </div>
        <div className="bars">
          {order.map(([p, i]) => (
            <div className="bar-row" key={i} style={{ gridTemplateColumns: "96px 1fr 46px" }}>
              <span className="name">{data.classes[i]}</span>
              <div className="bar-track"><div className={`fill${p === order[0][0] ? " hi" : ""}`} style={{ width: `${p * 100}%` }} /></div>
              <span className="val">{(p * 100).toFixed(0)}%</span>
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
