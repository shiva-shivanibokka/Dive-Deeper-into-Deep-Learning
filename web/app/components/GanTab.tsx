"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";

const ZDIM = 32;
const N = 16;
const tanhNorm = (v: number) => (v + 1) / 2;

export default function GanTab() {
  const refs = useRef<(HTMLCanvasElement | null)[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(false);

  const generate = useCallback(async () => {
    setBusy(true);
    try {
      setErr(false);
      const sess = await getSession("gan_generator.onnx");
      const noise = Float32Array.from({ length: N * ZDIM }, gaussian);
      const out = await sess.run({ z: new ort.Tensor("float32", noise, [N, ZDIM]) });
      const data = out.image.data as Float32Array;
      for (let n = 0; n < N; n++) {
        const c = refs.current[n];
        if (c) renderGray(c, data.slice(n * 784, (n + 1) * 784), 28, 28, tanhNorm);
      }
    } catch {
      setErr(true);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { generate(); }, [generate]);

  return (
    <div className="demo" style={{ justifyItems: "center" }}>
      <p className="section-label">Generated from random noise (none of these are real digits)</p>
      <div className="canvas-frame" style={{ padding: 12 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 108px)", gap: 10 }}>
          {Array.from({ length: N }).map((_, i) => (
            <canvas key={i} ref={(el) => { refs.current[i] = el; }} width={108} height={108}
              style={{ width: 108, height: 108, borderRadius: 8, background: "#000", imageRendering: "pixelated" }} />
          ))}
        </div>
      </div>
      <button className="btn primary" onClick={generate} disabled={busy}>{busy ? "generating…" : "🎲 Generate new batch"}</button>
      {err && <p className="note" style={{ color: "var(--bad)" }}>Couldn&apos;t run the model — try again.</p>}
    </div>
  );
}

function gaussian() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
