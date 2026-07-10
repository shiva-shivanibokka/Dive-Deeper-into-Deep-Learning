"use client";

import { useEffect, useRef, useState } from "react";

const EXAMPLES = [
  "This movie was an absolute masterpiece — I loved every minute of it.",
  "Worst purchase I've ever made. Broke after one day and support ignored me.",
];

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
      pipePromise = null;
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
    <div className="demo">
      <p className="callout" style={{ maxWidth: 720 }}>DistilBERT — the Notebook 04 architecture — running <strong>fully in your browser</strong> via Transformers.js (WebAssembly). No server. First run downloads the model (~65 MB), then it&apos;s instant.</p>
      <div className="results" style={{ maxWidth: 720 }}>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} style={{ width: "100%", resize: "vertical" }} />
        <div className="seg">
          {EXAMPLES.map((ex, i) => <button key={i} onClick={() => setText(ex)}>Example {i + 1}</button>)}
        </div>
        {error ? (
          <div className="callout" style={{ borderColor: "var(--bad)" }}>
            Couldn&apos;t load the model — check your connection.{" "}
            <button className="btn" onClick={() => { setStatus("loading DistilBERT…"); run(text); }}>Retry</button>
          </div>
        ) : status ? (
          <p className="note">{status} {progress > 0 && `${progress.toFixed(0)}%`}</p>
        ) : result && (
          <div className="readout" style={{ maxWidth: 380 }}>
            <div className="lbl">Sentiment</div>
            <div className="big" style={{ color: pos ? "var(--ok)" : "var(--bad)" }}>{pos ? "😊 Positive" : "😞 Negative"}</div>
            <div className="lbl" style={{ marginTop: ".3rem" }}>{(result.score * 100).toFixed(1)}% confident</div>
          </div>
        )}
      </div>
    </div>
  );
}
