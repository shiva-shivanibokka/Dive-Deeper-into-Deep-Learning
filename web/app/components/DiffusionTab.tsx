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
      const out = await sess.run({
        x: new ort.Tensor("float32", x, [1, 1, 28, 28]),
        t: new ort.Tensor("int64", BigInt64Array.from([BigInt(t)]), [1]),
      });
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
      await new Promise((r) => setTimeout(r, 16));
    }
    setRunning(false);
  }, [sched]);

  useEffect(() => { if (sched) sample(); }, [sched]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="demo" style={{ justifyItems: "center" }}>
      <p className="section-label">
        {running ? `denoising · step ${(sched?.steps ?? 0) - (step ?? 0)} / ${sched?.steps}` : "pure noise → digit"}
      </p>
      <div className="canvas-frame"><canvas ref={canvas} width={240} height={240} style={{ width: 240, height: 240 }} /></div>
      <button className="btn primary" onClick={sample} disabled={running || !sched}>
        {running ? "sampling…" : "🎲 Sample a new digit"}
      </button>
      <p className="callout" style={{ maxWidth: 520 }}>Diffusion runs the process backwards: start from static, and a U-Net predicts &amp; removes a little noise at each of the {sched?.steps ?? 40} steps until a digit emerges. This is the family behind Stable Diffusion &amp; DALL·E.</p>
    </div>
  );
}
