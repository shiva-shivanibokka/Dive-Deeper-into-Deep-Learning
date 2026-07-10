"use client";

import { useRef, useState, useCallback } from "react";
import { getSession, ort } from "../lib/onnx";
import DrawCanvas from "./DrawCanvas";

const EMPTY = Array(10).fill(0);

export default function CnnTab() {
  const [probs, setProbs] = useState<number[]>(EMPTY);
  const [drawn, setDrawn] = useState(false);
  const busy = useRef(false);

  const classify = useCallback(async (data: Float32Array | null) => {
    if (!data) { setProbs(EMPTY); setDrawn(false); return; }
    if (busy.current) return;
    busy.current = true;
    try {
      const sess = await getSession("mnist_cnn.onnx");
      const out = await sess.run({ image: new ort.Tensor("float32", data, [1, 1, 28, 28]) });
      setProbs(softmax(Array.from(out.logits.data as Float32Array)));
      setDrawn(true);
    } finally {
      busy.current = false;
    }
  }, []);

  const top = drawn ? probs.indexOf(Math.max(...probs)) : null;

  return (
    <div className="demo" style={{ display: "grid", gridTemplateColumns: "auto 1fr", alignItems: "start", gap: "2.5rem" }}>
      <div>
        <p className="section-label">Draw a digit 0–9</p>
        <DrawCanvas onResult={classify} size={340} />
      </div>
      {/* fixed layout so drawing never shifts the canvas */}
      <div className="results">
        <div className="readout">
          <div className="lbl">Predicted digit</div>
          <div className="big grad" style={{ minHeight: "1.05em" }}>{top === null ? "—" : top}</div>
        </div>
        <div>
          <p className="section-label">Class probabilities</p>
          <div className="bars">
            {probs.map((p, i) => (
              <div className="bar-row" key={i}>
                <span className="name" style={{ textAlign: "center" }}>{i}</span>
                <div className="bar-track"><div className={`fill${i === top ? " hi" : ""}`} style={{ width: `${p * 100}%` }} /></div>
                <span className="val">{drawn ? `${(p * 100).toFixed(0)}%` : "—"}</span>
              </div>
            ))}
          </div>
        </div>
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
