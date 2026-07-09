"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";

const Z = 8;
const tanhNorm = (v: number) => (v + 1) / 2;

export default function VaeTab() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [z, setZ] = useState<number[]>(Array(Z).fill(0));
  const [ready, setReady] = useState(false);

  const decode = useCallback(async (vec: number[]) => {
    const sess = await getSession("vae_decoder.onnx");
    const t = new ort.Tensor("float32", Float32Array.from(vec), [1, Z]);
    const out = await sess.run({ z: t });
    renderGray(canvas.current!, out.image.data, 28, 28, tanhNorm);
    setReady(true);
  }, []);

  useEffect(() => { decode(z); }, [z, decode]);

  const randomize = () => setZ(Array.from({ length: Z }, () => (Math.random() * 4 - 2)));
  const reset = () => setZ(Array(Z).fill(0));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: "1.5rem", alignItems: "start" }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: ".8rem" }}>
        <canvas ref={canvas} width={200} height={200}
          style={{ width: 200, height: 200, borderRadius: 10, border: "1px solid var(--border)", background: "#000", imageRendering: "pixelated" }} />
        {!ready && <span style={{ color: "var(--muted)", fontSize: ".8rem" }}>loading model…</span>}
        <div style={{ display: "flex", gap: ".5rem" }}>
          <button className="tab" onClick={randomize}>🎲 Randomize</button>
          <button className="tab" onClick={reset}>Reset</button>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: ".5rem 1rem" }}>
        {z.map((v, i) => (
          <label key={i} style={{ fontSize: ".78rem", color: "var(--muted)" }}>
            z{i + 1}: {v.toFixed(2)}
            <input type="range" min={-3} max={3} step={0.01} value={v}
              onChange={(e) => setZ((prev) => prev.map((p, j) => (j === i ? +e.target.value : p)))}
              style={{ width: "100%" }} />
          </label>
        ))}
      </div>
    </div>
  );
}
