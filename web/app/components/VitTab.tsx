"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";
import { softmax } from "../lib/preprocess";

type Sample = { label: string; pixels: number[] };
type Data = { classes: string[]; samples: Sample[] };
const BIG = 224; // 8 patches of 28px on screen (image is 4x4 patches of 7px)

export default function VitTab() {
  const [data, setData] = useState<Data | null>(null);
  const [sel, setSel] = useState(0);
  const [probs, setProbs] = useState<number[] | null>(null);
  const [showPatches, setShowPatches] = useState(true);
  const [err, setErr] = useState(false);
  const bigCanvas = useRef<HTMLCanvasElement>(null);
  const thumbs = useRef<(HTMLCanvasElement | null)[]>([]);

  useEffect(() => { fetch("/models/vit_samples.json").then((r) => r.json()).then(setData).catch(() => setErr(true)); }, []);

  useEffect(() => {
    if (!data) return;
    data.samples.forEach((s, i) => { const c = thumbs.current[i]; if (c) renderGray(c, Float32Array.from(s.pixels), 28, 28); });
  }, [data]);

  const drawPreview = useCallback((px: Float32Array, patches: boolean) => {
    const c = bigCanvas.current!;
    renderGray(c, px, 28, 28);
    if (patches) {
      const ctx = c.getContext("2d")!;
      ctx.strokeStyle = "rgba(34,211,238,0.7)"; ctx.lineWidth = 1;
      const step = BIG / 4; // 4x4 patches
      for (let i = 1; i < 4; i++) {
        ctx.beginPath(); ctx.moveTo(i * step, 0); ctx.lineTo(i * step, BIG); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i * step); ctx.lineTo(BIG, i * step); ctx.stroke();
      }
    }
  }, []);

  const classify = useCallback(async (d: Data, i: number, patches: boolean) => {
    try {
      const px = Float32Array.from(d.samples[i].pixels);
      drawPreview(px, patches);
      const sess = await getSession("vit.onnx");
      const out = await sess.run({ image: new ort.Tensor("float32", px, [1, 1, 28, 28]) });
      setProbs(softmax(Array.from(out.logits.data as Float32Array)));
    } catch { setErr(true); }
  }, [drawPreview]);

  useEffect(() => { if (data) classify(data, sel, showPatches); }, [data, sel, showPatches, classify]);

  if (!data) return <p className="note">{err ? "Couldn't load the model — check your connection." : "loading model…"}</p>;
  const order = probs ? probs.map((p, i) => [p, i] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, 3) : [];

  return (
    <div className="demo">
      <p className="callout">A Vision Transformer chops each image into a <strong>4×4 grid of patches</strong>, treats those 16 patches like words in a sentence, and runs the same self-attention a text Transformer uses. Toggle the grid to see the patches it reads.</p>
      <p className="section-label">Pick a Fashion-MNIST image</p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {data.samples.map((s, i) => (
          <canvas key={i} ref={(el) => { thumbs.current[i] = el; }} width={54} height={54} onClick={() => setSel(i)} title={s.label}
            style={{ width: 54, height: 54, borderRadius: 8, cursor: "pointer", background: "#000", imageRendering: "pixelated",
              border: `2px solid ${i === sel ? "var(--cyan)" : "var(--border)"}`, boxShadow: i === sel ? "0 0 14px -2px var(--cyan)" : "none" }} />
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "2.5rem", alignItems: "center" }}>
        <div style={{ textAlign: "center" }}>
          <div className="canvas-frame"><canvas ref={bigCanvas} width={BIG} height={BIG} style={{ width: BIG, height: BIG }} /></div>
          <div className="seg" style={{ justifyContent: "center", marginTop: ".6rem" }}>
            <button aria-pressed={showPatches} onClick={() => setShowPatches((v) => !v)}>Patch grid</button>
          </div>
          <div className="section-label" style={{ justifyContent: "center", margin: ".5rem 0 0" }}>True: {data.samples[sel].label}</div>
        </div>
        <div>
          <p className="section-label">Top predictions</p>
          <div className="bars">
            {order.map(([p, i]) => (
              <div className="bar-row" key={i} style={{ gridTemplateColumns: "100px 1fr 46px" }}>
                <span className="name">{data.classes[i]}</span>
                <div className="bar-track"><div className={`fill${p === order[0][0] ? " hi" : ""}`} style={{ width: `${p * 100}%` }} /></div>
                <span className="val">{(p * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
