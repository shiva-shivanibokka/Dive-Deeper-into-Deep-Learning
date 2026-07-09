"use client";

import { useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";
import DrawCanvas from "./DrawCanvas";

export default function AutoencoderTab() {
  const inC = useRef<HTMLCanvasElement>(null);
  const outC = useRef<HTMLCanvasElement>(null);
  const [err, setErr] = useState<number | null>(null);

  const run = useCallback(async (data: Float32Array | null) => {
    if (!data) { setErr(null); return; }
    renderGray(inC.current!, data, 28, 28);
    const sess = await getSession("autoencoder.onnx");
    const t = new ort.Tensor("float32", data, [1, 1, 28, 28]);
    const out = await sess.run({ image: t });
    const recon = out.recon.data as Float32Array;
    renderGray(outC.current!, recon, 28, 28);
    // reconstruction error = mean squared diff (this is exactly the anomaly-detection signal)
    let se = 0;
    for (let i = 0; i < 784; i++) se += (recon[i] - data[i]) ** 2;
    setErr(se / 784);
  }, []);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "1.5rem", alignItems: "center" }}>
      <DrawCanvas onResult={run} />
      <div style={{ display: "flex", gap: "1.5rem", alignItems: "center" }}>
        <Panel title="Your input" cref={inC} />
        <span style={{ fontSize: "1.5rem", color: "var(--muted)" }}>→</span>
        <Panel title="Reconstruction" cref={outC} />
        {err !== null && (
          <div style={{ marginLeft: "1rem" }}>
            <div style={{ color: "var(--muted)", fontSize: ".8rem" }}>Reconstruction error</div>
            <div style={{ fontSize: "1.4rem", fontWeight: 700, color: err > 0.05 ? "#f59e0b" : "var(--accent)" }}>{err.toFixed(4)}</div>
            <div style={{ color: "var(--muted)", fontSize: ".72rem", maxWidth: 180 }}>
              High error = the input looks unlike the digits the model learned (anomaly signal).
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Panel({ title, cref }: { title: string; cref: React.RefObject<HTMLCanvasElement | null> }) {
  return (
    <div style={{ textAlign: "center" }}>
      <canvas ref={cref} width={140} height={140}
        style={{ width: 140, height: 140, borderRadius: 8, border: "1px solid var(--border)", background: "#000", imageRendering: "pixelated" }} />
      <div style={{ color: "var(--muted)", fontSize: ".75rem", marginTop: ".3rem" }}>{title}</div>
    </div>
  );
}
