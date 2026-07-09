"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";

const ZDIM = 32;
const N = 16; // 4x4 grid
const tanhNorm = (v: number) => (v + 1) / 2;

export default function GanTab() {
  const refs = useRef<(HTMLCanvasElement | null)[]>([]);
  const [busy, setBusy] = useState(false);

  const generate = useCallback(async () => {
    setBusy(true);
    const sess = await getSession("gan_generator.onnx");
    const noise = Float32Array.from({ length: N * ZDIM }, () => gaussian());
    const t = new ort.Tensor("float32", noise, [N, ZDIM]);
    const out = await sess.run({ z: t });
    const data = out.image.data as Float32Array;
    for (let n = 0; n < N; n++) {
      const c = refs.current[n];
      if (c) renderGray(c, data.slice(n * 784, (n + 1) * 784), 28, 28, tanhNorm);
    }
    setBusy(false);
  }, []);

  useEffect(() => { generate(); }, [generate]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 64px)", gap: 6 }}>
        {Array.from({ length: N }).map((_, i) => (
          <canvas key={i} ref={(el) => { refs.current[i] = el; }} width={64} height={64}
            style={{ width: 64, height: 64, borderRadius: 6, border: "1px solid var(--border)", background: "#000", imageRendering: "pixelated" }} />
        ))}
      </div>
      <button className="tab" onClick={generate} disabled={busy}>{busy ? "generating…" : "🎲 Generate new batch"}</button>
    </div>
  );
}

// Box-Muller for standard-normal noise (GANs are trained on N(0,1)).
function gaussian() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
