"use client";

import { useEffect, useRef, useState } from "react";

const EXAMPLES = [
  "This movie was an absolute masterpiece — I loved every minute of it.",
  "Worst purchase I've ever made. Broke after one day and support ignored me.",
];

// Lazily build the transformers.js sentiment pipeline (downloads DistilBERT weights once, ~65MB).
let pipePromise: Promise<(t: string) => Promise<{ label: string; score: number }[]>> | null = null;
async function getPipe(onProgress: (p: number) => void) {
  if (!pipePromise) {
    pipePromise = (async () => {
      const { pipeline, env } = await import("@xenova/transformers");
      env.allowLocalModels = false;
      const pipe = await pipeline("sentiment-analysis", "Xenova/distilbert-base-uncased-finetuned-sst-2-english", {
        progress_callback: (d: { status: string; progress?: number }) => {
          if (d.status === "progress" && d.progress != null) onProgress(d.progress);
        },
      });
      return (t: string) => pipe(t) as Promise<{ label: string; score: number }[]>;
    })();
  }
  return pipePromise;
}

export default function BertTab() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [result, setResult] = useState<{ label: string; score: number } | null>(null);
  const [status, setStatus] = useState("loading DistilBERT (first run downloads the model)…");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(false);
  const ready = useRef(false);

  const run = async (t: string) => {
    try {
      setError(false);
      const pipe = await getPipe(setProgress);
      ready.current = true;
      setStatus("");
      const out = await pipe(t);
      setResult(out[0]);
    } catch {
      pipePromise = null; // let a retry re-fetch the model
      ready.current = false;
      setStatus("");
      setError(true);
    }
  };

  useEffect(() => { run(text); /* eslint-disable-next-line */ }, []);
  useEffect(() => {
    if (!ready.current) return;
    const id = setTimeout(() => run(text), 400);
    return () => clearTimeout(id);
  }, [text]);

  const pos = result?.label === "POSITIVE";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <p style={{ color: "var(--muted)", fontSize: ".8rem", margin: 0 }}>
        DistilBERT (the Notebook 04 architecture) running <strong>fully in your browser</strong> via transformers.js — WASM, no server.
      </p>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3}
        style={{ width: "100%", background: "var(--panel-2)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 8, padding: ".7rem", fontSize: ".9rem", resize: "vertical" }} />
      <div style={{ display: "flex", gap: ".5rem" }}>
        {EXAMPLES.map((ex, i) => (
          <button key={i} className="tab" onClick={() => setText(ex)} style={{ fontSize: ".75rem" }}>Example {i + 1}</button>
        ))}
      </div>
      {error ? (
        <div style={{ color: "#f87171", fontSize: ".85rem" }}>
          Couldn&apos;t load the model — check your connection.{" "}
          <button className="tab" onClick={() => { setStatus("loading DistilBERT…"); run(text); }}>Retry</button>
        </div>
      ) : status ? (
        <div style={{ color: "var(--muted)", fontSize: ".85rem" }}>
          {status} {progress > 0 && `${progress.toFixed(0)}%`}
        </div>
      ) : result && (
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <div style={{ fontSize: "2rem", fontWeight: 800, color: pos ? "var(--accent)" : "#f87171" }}>
            {pos ? "😊 Positive" : "😞 Negative"}
          </div>
          <div style={{ color: "var(--muted)", fontSize: ".9rem" }}>{(result.score * 100).toFixed(1)}% confident</div>
        </div>
      )}
    </div>
  );
}
