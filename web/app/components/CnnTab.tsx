"use client";

import { useState, useCallback } from "react";
import { getSession, ort } from "../lib/onnx";
import DrawCanvas from "./DrawCanvas";

export default function CnnTab() {
  const [probs, setProbs] = useState<number[] | null>(null);

  const classify = useCallback(async (data: Float32Array | null) => {
    if (!data) { setProbs(null); return; }
    const sess = await getSession("mnist_cnn.onnx");
    const out = await sess.run({ image: new ort.Tensor("float32", data, [1, 1, 28, 28]) });
    setProbs(softmax(Array.from(out.logits.data as Float32Array)));
  }, []);

  const top = probs ? probs.indexOf(Math.max(...probs)) : null;

  return (
    <div className="demo" style={{ gridTemplateColumns: "auto 1fr", display: "grid", alignItems: "center", gap: "2rem" }}>
      <div>
        <p className="section-label">Draw a digit 0–9</p>
        <DrawCanvas onResult={classify} />
      </div>
      <div className="results">
        {top === null ? (
          <p className="note">Draw a digit on the pad — the CNN classifies it live as you draw. Inputs are centered and scaled to match MNIST before inference.</p>
        ) : (
          <>
            <div className="readout">
              <div className="lbl">Predicted digit</div>
              <div className="big grad">{top}</div>
            </div>
            <div>
              <p className="section-label">Class probabilities</p>
              <div className="bars">
                {probs!.map((p, i) => (
                  <div className="bar-row" key={i}>
                    <span className="name" style={{ textAlign: "center" }}>{i}</span>
                    <div className="bar-track"><div className={`fill${i === top ? " hi" : ""}`} style={{ width: `${p * 100}%` }} /></div>
                    <span className="val">{(p * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
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
