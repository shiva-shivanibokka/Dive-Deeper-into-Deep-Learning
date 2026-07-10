"use client";

import { useEffect, useRef, useState } from "react";

const MODEL = "Xenova/twitter-roberta-base-sentiment-latest"; // negative / neutral / positive
const EMOJI: Record<string, string> = { positive: "😊", neutral: "😐", negative: "😞" };
const EXAMPLES = [
  "This is the best day of my life, I can't stop smiling!",
  "I'm so hungry, I could eat a whole pizza right now.",
  "Worst purchase ever — it broke after one day.",
  "The meeting is scheduled for 3pm on Thursday.",
];

type Res = { label: string; score: number };
let pipePromise: Promise<(t: string) => Promise<Res[]>> | null = null;
async function getPipe(onProgress: (p: number) => void) {
  if (!pipePromise) {
    pipePromise = (async () => {
      const { pipeline, env } = await import("@xenova/transformers");
      env.allowLocalModels = false;
      const pipe = await pipeline("text-classification", MODEL, {
        progress_callback: (d: { status: string; progress?: number }) => {
          if (d.status === "progress" && d.progress != null) onProgress(d.progress);
        },
      });
      return (t: string) => pipe(t, { topk: 3 }) as Promise<Res[]>;
    })();
  }
  return pipePromise;
}

export default function BertTab() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [res, setRes] = useState<Res[] | null>(null);
  const [status, setStatus] = useState("loading the emotion model (first run downloads it)…");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(false);
  const ready = useRef(false);

  const run = async (t: string) => {
    try {
      setError(false);
      const pipe = await getPipe(setProgress);
      ready.current = true;
      setStatus("");
      setRes(await pipe(t));
    } catch {
      pipePromise = null; ready.current = false; setStatus(""); setError(true);
    }
  };

  useEffect(() => { run(text); /* eslint-disable-next-line */ }, []);
  useEffect(() => {
    if (!ready.current) return;
    const id = setTimeout(() => run(text), 400);
    return () => clearTimeout(id);
  }, [text]);

  const sorted = res ? [...res].sort((a, b) => b.score - a.score) : [];
  const top = sorted[0];

  return (
    <div className="demo">
      <p className="callout">A RoBERTa model (BERT family — the Notebook 04 architecture) running <strong>fully in your browser</strong> via Transformers.js — no server. It reads your text and scores it as <strong>positive, neutral, or negative</strong> (so &ldquo;I&apos;m hungry&rdquo; reads as neutral, not negative). First run downloads the model, then it&apos;s instant.</p>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} style={{ width: "100%", resize: "vertical" }} />
      <div className="seg">
        {EXAMPLES.map((ex, i) => <button key={i} onClick={() => setText(ex)}>Example {i + 1}</button>)}
      </div>
      {error ? (
        <div className="callout" style={{ borderColor: "var(--bad)" }}>
          Couldn&apos;t load the model — check your connection.{" "}
          <button className="btn" onClick={() => { setStatus("loading…"); run(text); }}>Retry</button>
        </div>
      ) : status ? (
        <p className="note">{status} {progress > 0 && `${progress.toFixed(0)}%`}</p>
      ) : top && (
        <div>
          <div className="readout" style={{ marginBottom: "1.1rem" }}>
            <div className="lbl">Sentiment</div>
            <div className="big grad" style={{ textTransform: "capitalize" }}>{EMOJI[top.label] ?? "•"} {top.label}</div>
          </div>
          <p className="section-label">Breakdown</p>
          <div className="bars">
            {sorted.map((r) => (
              <div className="bar-row" key={r.label} style={{ gridTemplateColumns: "130px 1fr 46px" }}>
                <span className="name" style={{ textTransform: "capitalize" }}>{EMOJI[r.label] ?? ""} {r.label}</span>
                <div className="bar-track"><div className={`fill${r.label === top.label ? " hi" : ""}`} style={{ width: `${r.score * 100}%` }} /></div>
                <span className="val">{(r.score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
