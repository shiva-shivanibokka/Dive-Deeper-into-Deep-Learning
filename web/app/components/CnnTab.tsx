"use client";

import { useState, useCallback } from "react";
import { getSession, ort } from "../lib/onnx";
import DrawCanvas from "./DrawCanvas";

export default function CnnTab() {
  const [probs, setProbs] = useState<number[] | null>(null);

  const classify = useCallback(async (data: Float32Array | null) => {
    if (!data) { setProbs(null); return; }
    const sess = await getSession("mnist_cnn.onnx");
    const t = new ort.Tensor("float32", data, [1, 1, 28, 28]);
    const out = await sess.run({ image: t });
    setProbs(softmax(Array.from(out.logits.data as Float32Array)));
  }, []);

  const top = probs ? probs.indexOf(Math.max(...probs)) : null;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "1.5rem", alignItems: "center" }}>
      <DrawCanvas onResult={classify} />
      <div>
        {top === null ? (
          <p style={{ color: "var(--muted)" }}>Draw a digit (0–9) on the pad.</p>
        ) : (
          <>
            <div style={{ fontSize: "2.2rem", fontWeight: 700, marginBottom: ".6rem" }}>
              Prediction: <span style={{ color: "var(--accent)" }}>{top}</span>
            </div>
            {probs!.map((p, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: ".5rem", fontSize: ".8rem" }}>
                <span style={{ width: 14, color: "var(--muted)" }}>{i}</span>
                <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 4, height: 12 }}>
                  <div style={{ width: `${p * 100}%`, height: "100%", background: i === top ? "var(--accent)" : "var(--accent-2)", borderRadius: 4 }} />
                </div>
                <span style={{ width: 42, textAlign: "right", color: "var(--muted)" }}>{(p * 100).toFixed(1)}%</span>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

function softmax(a: number[]) {
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
}
