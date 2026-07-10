"use client";

import { useEffect, useRef, useState } from "react";

// 28-class GoEmotions model (multi-label), self-hosted under /public/models/goemotions
// in the standard Transformers.js layout so it can never fail to load from a third party.
const MODEL = "goemotions";
const EMOJI: Record<string, string> = {
  admiration: "🤩", amusement: "😄", anger: "😠", annoyance: "😤", approval: "👍",
  caring: "🤗", confusion: "😕", curiosity: "🤔", desire: "😍", disappointment: "😞",
  disapproval: "👎", disgust: "🤢", embarrassment: "😳", excitement: "🤸", fear: "😨",
  gratitude: "🙏", grief: "😢", joy: "😊", love: "❤️", nervousness: "😬",
  optimism: "🙂", pride: "😌", realization: "💡", relief: "😮‍💨", remorse: "😔",
  sadness: "😢", surprise: "😲", neutral: "😐",
};
const EXAMPLES = [
  "This is the best day of my life, I can't stop smiling!",
  "I'm so hungry, I could eat a whole pizza right now.",
  "Worst purchase ever — it broke after one day.",
  "Wait, I think I finally understand how this works!",
];

type Res = { label: string; score: number };
let pipePromise: Promise<(t: string) => Promise<Res[]>> | null = null;
async function getPipe(onProgress: (p: number) => void) {
  if (!pipePromise) {
    pipePromise = (async () => {
      const { pipeline, env } = await import("@xenova/transformers");
      env.allowLocalModels = true;      // load from our own /public, not Hugging Face
      env.localModelPath = "/models/";  // -> /models/goemotions/{config,onnx/model_quantized.onnx}
      env.allowRemoteModels = false;
      const pipe = await pipeline("text-classification", MODEL, {
        progress_callback: (d: { status: string; progress?: number }) => {
          if (d.status === "progress" && d.progress != null) onProgress(d.progress);
        },
      });
      return (t: string) => pipe(t, { topk: 6 }) as Promise<Res[]>;
    })();
  }
  return pipePromise;
}

export default function BertTab() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [res, setRes] = useState<Res[] | null>(null);
  const [status, setStatus] = useState("loading the emotion model (first run downloads ~30 MB)…");
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
      <p className="callout">A DistilBERT emotion model (BERT family — the Notebook 04 architecture) running <strong>fully in your browser</strong> via Transformers.js — no server. It scores your text across <strong>28 fine-grained emotions</strong> (Google&apos;s GoEmotions taxonomy: joy, admiration, curiosity, annoyance, grief…) and shows the strongest few — so you see a nuanced blend, not just &ldquo;positive/negative&rdquo;. First run downloads the model, then it&apos;s instant.</p>
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
            <div className="lbl">Top emotion</div>
            <div className="big" style={{ textTransform: "capitalize", display: "flex", alignItems: "center", justifyContent: "center", gap: ".4rem" }}>
              <span>{EMOJI[top.label] ?? "•"}</span>
              <span style={{ background: "var(--grad)", WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>{top.label}</span>
            </div>
          </div>
          <p className="section-label">Top 6 of 28 emotions</p>
          <div className="bars">
            {sorted.map((r) => (
              <div className="bar-row" key={r.label} style={{ gridTemplateColumns: "160px 1fr 46px" }}>
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
