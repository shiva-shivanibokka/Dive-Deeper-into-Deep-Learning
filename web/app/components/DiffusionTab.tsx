"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";

type Sched = { steps: number; betas: number[]; alphas: number[]; abars: number[] };
const tanhNorm = (v: number) => (v + 1) / 2;
const N = 784;

function randn(n: number) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    a[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return a;
}

export default function DiffusionTab() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [sched, setSched] = useState<Sched | null>(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState<number | null>(null);

  useEffect(() => {
    fetch("/models/diffusion_schedule.json").then((r) => r.json()).then(setSched);
  }, []);

  const sample = useCallback(async () => {
    if (!sched) return;
    setRunning(true);
    const sess = await getSession("diffusion_unet.onnx");
    let x = randn(N);
    for (let t = sched.steps - 1; t >= 0; t--) {
      const xt = new ort.Tensor("float32", x, [1, 1, 28, 28]);
      const tt = new ort.Tensor("int64", BigInt64Array.from([BigInt(t)]), [1]);
      const out = await sess.run({ x: xt, t: tt });
      const eps = out.noise.data as Float32Array;
      const coef = sched.betas[t] / Math.sqrt(1 - sched.abars[t]);
      const sa = Math.sqrt(sched.alphas[t]);
      const noise = t > 0 ? randn(N) : null;
      const sb = Math.sqrt(sched.betas[t]);
      const nx = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        const mean = (x[i] - coef * eps[i]) / sa;
        nx[i] = noise ? mean + sb * noise[i] : mean;
      }
      x = nx;
      renderGray(canvas.current!, x, 28, 28, tanhNorm);
      setStep(t);
      await new Promise((r) => setTimeout(r, 16)); // let the browser paint the frame
    }
    setRunning(false);
  }, [sched]);

  useEffect(() => { if (sched) sample(); }, [sched]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem" }}>
      <canvas ref={canvas} width={220} height={220}
        style={{ width: 220, height: 220, borderRadius: 10, border: "1px solid var(--border)", background: "#000", imageRendering: "pixelated" }} />
      <div style={{ color: "var(--muted)", fontSize: ".8rem", minHeight: 18 }}>
        {running ? `denoising… step ${(sched?.steps ?? 0) - (step ?? 0)} / ${sched?.steps}` : "done — pure noise → digit"}
      </div>
      <button className="tab" onClick={sample} disabled={running || !sched}>
        {running ? "sampling…" : "🎲 Sample a new digit"}
      </button>
    </div>
  );
}
