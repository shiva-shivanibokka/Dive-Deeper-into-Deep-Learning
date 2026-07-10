"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";

const Z = 8;
const tanhNorm = (v: number) => (v + 1) / 2;

export default function VaeTab() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [z, setZ] = useState<number[]>(Array(Z).fill(0));

  const decode = useCallback(async (vec: number[]) => {
    const sess = await getSession("vae_decoder.onnx");
    const out = await sess.run({ z: new ort.Tensor("float32", Float32Array.from(vec), [1, Z]) });
    renderGray(canvas.current!, out.image.data, 28, 28, tanhNorm);
  }, []);

  useEffect(() => { decode(z); }, [z, decode]);

  const randomize = () => setZ(Array.from({ length: Z }, () => Math.random() * 4 - 2));

  return (
    <div className="demo" style={{ gridTemplateColumns: "auto 1fr", display: "grid", alignItems: "start", gap: "2rem" }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: ".9rem" }}>
        <div className="canvas-frame"><canvas ref={canvas} width={220} height={220} style={{ width: 220, height: 220 }} /></div>
        <div className="seg">
          <button onClick={randomize}>🎲 Randomize</button>
          <button onClick={() => setZ(Array(Z).fill(0))}>Reset</button>
        </div>
      </div>
      <div className="results">
        <p className="section-label">Latent vector (8 dimensions)</p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: ".7rem 1.5rem" }}>
          {z.map((v, i) => (
            <div className="field" key={i}>
              <label><span>z{i + 1}</span> <b>{v.toFixed(2)}</b></label>
              <input type="range" min={-3} max={3} step={0.01} value={v}
                onChange={(e) => setZ((prev) => prev.map((p, j) => (j === i ? +e.target.value : p)))} />
            </div>
          ))}
        </div>
        <p className="callout">Each slider moves through the VAE&apos;s learned latent space. Small moves morph the digit smoothly — proof the space is continuous and meaningful, not just memorized images.</p>
      </div>
    </div>
  );
}
