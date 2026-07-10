"use client";

import { useEffect, useState, useCallback } from "react";
import { getSession, ort } from "../lib/onnx";
import { softmax, tokenize, encodeTokens } from "../lib/preprocess";

type Vocab = { word2idx: Record<string, number>; maxlen: number; labels: string[] };
const EXAMPLES = [
  "The goalie made an incredible save in overtime to win the playoff game.",
  "The new turbocharged engine gets great mileage but the transmission is rough.",
  "The patient was prescribed antibiotics after the doctor diagnosed an infection.",
  "The spacecraft entered orbit around the moon after a three-day journey.",
  "I rendered the 3D model with ray tracing and exported the image as a PNG.",
  "Peace negotiations in the region stalled again over the disputed border.",
];

export default function LstmTab() {
  const [vocab, setVocab] = useState<Vocab | null>(null);
  const [text, setText] = useState(EXAMPLES[0]);
  const [probs, setProbs] = useState<number[] | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => { fetch("/models/lstm_vocab.json").then((r) => r.json()).then(setVocab).catch(() => setErr(true)); }, []);

  const classify = useCallback(async (v: Vocab, t: string) => {
    if (tokenize(t).length === 0) { setProbs(null); return; } // nothing to classify
    try {
      const ids = encodeTokens(t, v.word2idx, v.maxlen);
      const sess = await getSession("lstm_text.onnx");
      const out = await sess.run({ tokens: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, v.maxlen]) });
      setProbs(softmax(Array.from(out.logits.data as Float32Array)));
    } catch { setErr(true); }
  }, []);

  useEffect(() => {
    if (!vocab) return;
    const id = setTimeout(() => classify(vocab, text), 200);
    return () => clearTimeout(id);
  }, [vocab, text, classify]);

  if (!vocab) return <p className="note">{err ? "Couldn't load the model — check your connection." : "loading model…"}</p>;
  const top = probs ? probs.indexOf(Math.max(...probs)) : null;
  const empty = tokenize(text).length === 0;

  return (
    <div className="demo">
      <p className="callout">
        A bi-directional LSTM trained on <strong>6 newsgroup topics</strong>: {vocab.labels.join(" · ")}.
        Type anything and it sorts your text into the <em>closest</em> of these six — it only knows these topics, so it will always pick one (that&apos;s the honest limit of a fixed-class classifier).
      </p>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} style={{ width: "100%", resize: "vertical" }}
        placeholder="Type a sentence about sports, cars, medicine, space, graphics, or politics…" />
      <div className="seg">
        {EXAMPLES.map((_, i) => <button key={i} onClick={() => setText(EXAMPLES[i])}>{vocab.labels[i]}</button>)}
      </div>
      {empty ? (
        <p className="note">Type a sentence above and its topic will appear here.</p>
      ) : probs && top !== null && (
        <div>
          <p className="section-label">Predicted topic</p>
          <div className="bars">
            {vocab.labels.map((lab, i) => (
              <div className="bar-row" key={i} style={{ gridTemplateColumns: "150px 1fr 46px" }}>
                <span className="name">{lab}</span>
                <div className="bar-track"><div className={`fill${i === top ? " hi" : ""}`} style={{ width: `${probs[i] * 100}%` }} /></div>
                <span className="val">{(probs[i] * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
