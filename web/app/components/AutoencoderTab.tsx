"use client";

import { useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";
import DrawCanvas from "./DrawCanvas";

export default function AutoencoderTab() {
  const inC = useRef<HTMLCanvasElement>(null);
  const outC = useRef<HTMLCanvasElement>(null);
  const [err, setErr] = useState<number | null>(null);
  const busy = useRef(false);

  const clearCanvas = (c: HTMLCanvasElement | null) => {
    if (!c) return;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, c.width, c.height);
  };

  const run = useCallback(async (data: Float32Array | null) => {
    if (!data) {
      setErr(null);
      clearCanvas(inC.current);
      clearCanvas(outC.current);
      return;
    }
    if (busy.current) return; // drop overlapping runs while drawing fast
    busy.current = true;
    try {
      renderGray(inC.current!, data, 28, 28);
      const sess = await getSession("autoencoder.onnx");
      const out = await sess.run({ image: new ort.Tensor("float32", data, [1, 1, 28, 28]) });
      const recon = out.recon.data as Float32Array;
      renderGray(outC.current!, recon, 28, 28);
      let se = 0;
      for (let i = 0; i < 784; i++) se += (recon[i] - data[i]) ** 2;
      setErr(se / 784);
    } finally {
      busy.current = false;
    }
  }, []);

  const anomalous = err !== null && err > 0.045;

  return (
    <div className="demo" style={{ gridTemplateColumns: "auto 1fr", display: "grid", alignItems: "center", gap: "2rem" }}>
      <div>
        <p className="section-label">Draw a digit</p>
        <DrawCanvas onResult={run} />
      </div>
      <div className="results">
        {/* canvases are ALWAYS mounted so the refs are valid on the first stroke */}
        <div style={{ display: "flex", gap: "1.5rem", alignItems: "center", flexWrap: "wrap" }}>
          <Panel title="Input" cref={inC} />
          <span style={{ fontSize: "1.6rem", color: "var(--muted)" }}>→</span>
          <Panel title="Reconstruction" cref={outC} />
        </div>
        {err === null ? (
          <p className="note">Draw a digit on the pad — the autoencoder squeezes it through a 16-number bottleneck and rebuilds it on the right. A real digit rebuilds cleanly (low error); a scribble it hasn&apos;t seen rebuilds poorly (high error) — that gap is how autoencoders flag anomalies.</p>
        ) : (
          <div className="readout" style={{ maxWidth: 340 }}>
            <div className="lbl">Reconstruction error {anomalous ? "· looks anomalous" : "· looks like a digit"}</div>
            <div className="big" style={{ color: anomalous ? "var(--amber)" : "var(--ok)" }}>{err.toFixed(4)}</div>
          </div>
        )}
      </div>
    </div>
  );
}

function Panel({ title, cref }: { title: string; cref: React.RefObject<HTMLCanvasElement | null> }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div className="canvas-frame"><canvas ref={cref} width={120} height={120} style={{ width: 120, height: 120, background: "#000" }} /></div>
      <div className="section-label" style={{ justifyContent: "center", margin: ".45rem 0 0" }}>{title}</div>
    </div>
  );
}
