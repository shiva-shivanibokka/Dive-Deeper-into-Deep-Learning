"use client";

import { useRef, useState, useCallback } from "react";
import { getSession, ort, renderGray } from "../lib/onnx";
import DrawCanvas from "./DrawCanvas";

export default function AutoencoderTab() {
  const inC = useRef<HTMLCanvasElement>(null);
  const outC = useRef<HTMLCanvasElement>(null);
  const [err, setErr] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);
  const busy = useRef(false);
  const pending = useRef<Float32Array | null>(null);

  const clearCanvas = (c: HTMLCanvasElement | null) => {
    if (!c) return;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, c.width, c.height);
  };

  const run = useCallback(async (data: Float32Array | null) => {
    if (!data) {
      setErr(null);
      pending.current = null;
      clearCanvas(inC.current);
      clearCanvas(outC.current);
      return;
    }
    // Keep only the latest frame while a run is in flight, then run it once free,
    // so the finished drawing isn't left showing a mid-stroke reconstruction.
    if (busy.current) { pending.current = data; return; }
    busy.current = true;
    try {
      setFailed(false);
      if (inC.current) renderGray(inC.current, data, 28, 28);
      const sess = await getSession("autoencoder.onnx");
      const out = await sess.run({ image: new ort.Tensor("float32", data, [1, 1, 28, 28]) });
      const recon = out.recon.data as Float32Array;
      if (outC.current) renderGray(outC.current, recon, 28, 28);
      let se = 0;
      for (let i = 0; i < 784; i++) se += (recon[i] - data[i]) ** 2;
      setErr(se / 784);
    } catch {
      setFailed(true);
    } finally {
      busy.current = false;
      const next = pending.current;
      if (next) { pending.current = null; run(next); } // trailing run on the latest frame
    }
  }, []);

  const anomalous = err !== null && err > 0.045;

  return (
    <div className="demo" style={{ gridTemplateColumns: "auto 1fr", display: "grid", alignItems: "start", gap: "2.5rem" }}>
      <div>
        <p className="section-label">Draw a digit</p>
        <DrawCanvas onResult={run} size={300} />
      </div>
      <div className="results" style={{ gap: "1.5rem" }}>
        {/* canvases are ALWAYS mounted so the refs are valid on the first stroke */}
        <div style={{ display: "flex", gap: "1.5rem", alignItems: "center", justifyContent: "center" }}>
          <Panel title="Input" cref={inC} />
          <span style={{ fontSize: "2rem", color: "var(--muted)" }}>→</span>
          <Panel title="Reconstruction" cref={outC} />
        </div>
        {failed && <p className="note" style={{ color: "var(--bad)" }}>Couldn&apos;t run the model — try drawing again.</p>}
        {err === null ? (
          <p className="note">Draw a digit on the pad — the autoencoder squeezes it through a 16-number bottleneck and rebuilds it on the right. A real digit rebuilds cleanly (low error); a scribble it hasn&apos;t seen rebuilds poorly (high error) — that gap is how autoencoders flag anomalies.</p>
        ) : (
          <div className="readout">
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
      <div className="canvas-frame"><canvas ref={cref} width={176} height={176} style={{ width: 176, height: 176, background: "#000" }} /></div>
      <div className="section-label" style={{ justifyContent: "center", margin: ".5rem 0 0" }}>{title}</div>
    </div>
  );
}
