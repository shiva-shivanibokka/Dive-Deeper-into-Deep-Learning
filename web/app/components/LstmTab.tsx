"use client";

import { useEffect, useState, useCallback } from "react";
import { getSession, ort } from "../lib/onnx";

type Vocab = { word2idx: Record<string, number>; maxlen: number; labels: string[] };
const EXAMPLES = [
  "The goalie made an incredible save in overtime to win the playoff game.",
  "The patient was prescribed antibiotics after the doctor diagnosed an infection.",
];

export default function LstmTab() {
  const [vocab, setVocab] = useState<Vocab | null>(null);
  const [text, setText] = useState(EXAMPLES[0]);
  const [probs, setProbs] = useState<number[] | null>(null);

  useEffect(() => {
    fetch("/models/lstm_vocab.json").then((r) => r.json()).then(setVocab);
  }, []);

  const classify = useCallback(async (v: Vocab, t: string) => {
    const words = (t.toLowerCase().match(/[a-z]+/g) ?? []).slice(0, v.maxlen);
    const ids = words.map((w) => v.word2idx[w] ?? 1);
    while (ids.length < v.maxlen) ids.push(0);
    const sess = await getSession("lstm_text.onnx");
    const out = await sess.run({ tokens: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, v.maxlen]) });
    setProbs(softmax(Array.from(out.logits.data as Float32Array)));
  }, []);

  useEffect(() => {
    if (!vocab) return;
    const id = setTimeout(() => classify(vocab, text), 200);
    return () => clearTimeout(id);
  }, [vocab, text, classify]);

  if (!vocab) return <p style={{ color: "var(--muted)" }}>loading model…</p>;
  const top = probs ? probs.indexOf(Math.max(...probs)) : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3}
        placeholder="Type a sentence about hockey or medicine…"
        style={{ width: "100%", background: "var(--panel-2)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 8, padding: ".7rem", fontSize: ".9rem", resize: "vertical" }} />
      <div style={{ display: "flex", gap: ".5rem" }}>
        {EXAMPLES.map((ex, i) => (
          <button key={i} className="tab" onClick={() => setText(ex)} style={{ fontSize: ".75rem" }}>
            Example {i + 1}
          </button>
        ))}
      </div>
      {probs && top !== null && (
        <div>
          <div style={{ fontSize: "1.3rem", fontWeight: 700, marginBottom: ".6rem" }}>
            → <span style={{ color: "var(--accent)" }}>{vocab.labels[top]}</span>
          </div>
          {vocab.labels.map((lab, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: ".5rem", fontSize: ".82rem", marginBottom: 3 }}>
              <span style={{ width: 130, color: "var(--muted)" }}>{lab}</span>
              <div style={{ flex: 1, background: "var(--panel-2)", borderRadius: 4, height: 12 }}>
                <div style={{ width: `${probs[i] * 100}%`, height: "100%", background: i === top ? "var(--accent)" : "var(--accent-2)", borderRadius: 4 }} />
              </div>
              <span style={{ width: 44, textAlign: "right", color: "var(--muted)" }}>{(probs[i] * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function softmax(a: number[]) {
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
}
