"use client";

import { useEffect, useState, useCallback } from "react";
import { getSession, ort } from "../lib/onnx";

type Meta = {
  features: string[];
  numeric: string[];
  mean: number[];
  std: number[];
  ranges: Record<string, [number, number]>;
  defaults: Record<string, number>;
};

const LABELS: Record<string, string> = {
  age: "Age",
  "education-num": "Education level (1–16)",
  "hours-per-week": "Hours worked / week",
  "capital-gain": "Capital gains ($)",
};

export default function MlpTab() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [vals, setVals] = useState<Record<string, number>>({});
  const [prob, setProb] = useState<number | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    fetch("/models/tabular_meta.json").then((r) => r.json())
      .then((m: Meta) => { setMeta(m); setVals({ ...m.defaults }); })
      .catch(() => setErr(true));
  }, []);

  const predict = useCallback(async (m: Meta, v: Record<string, number>) => {
    try {
      const vec = m.features.map((f) => {
        const ni = m.numeric.indexOf(f);
        return ni >= 0 ? (v[f] - m.mean[ni]) / m.std[ni] : v[f];
      });
      const sess = await getSession("tabular_mlp.onnx");
      const out = await sess.run({ features: new ort.Tensor("float32", Float32Array.from(vec), [1, vec.length]) });
      const logit = Number((out.logit.data as Float32Array)[0]);
      setProb(1 / (1 + Math.exp(-logit)));
    } catch { setErr(true); }
  }, []);

  useEffect(() => { if (meta && Object.keys(vals).length) predict(meta, vals); }, [meta, vals, predict]);

  if (!meta) return <p className="note">{err ? "Couldn't load the model — check your connection." : "loading model…"}</p>;
  const set = (k: string, n: number) => setVals((p) => ({ ...p, [k]: n }));
  const pct = prob == null ? 0 : prob * 100;

  return (
    <div className="demo" style={{ gridTemplateColumns: "1fr 300px", display: "grid", alignItems: "center", gap: "2.5rem" }}>
      <div className="results" style={{ gap: "1rem" }}>
        {meta.numeric.map((f) => (
          <div className="field" key={f}>
            <label><span>{LABELS[f] ?? f}</span> <b>{vals[f]}</b></label>
            <input type="range" min={meta.ranges[f][0]} max={meta.ranges[f][1]} value={vals[f] ?? 0} onChange={(e) => set(f, +e.target.value)} />
          </div>
        ))}
        <div className="seg" style={{ marginTop: ".3rem" }}>
          <button aria-pressed={vals.is_married === 1} onClick={() => set("is_married", vals.is_married === 1 ? 0 : 1)}>Married: {vals.is_married === 1 ? "Yes" : "No"}</button>
          <button aria-pressed={vals.sex_is_male === 1} onClick={() => set("sex_is_male", vals.sex_is_male === 1 ? 0 : 1)}>Male: {vals.sex_is_male === 1 ? "Yes" : "No"}</button>
        </div>
      </div>
      <div className="readout">
        <div className="lbl">Probability of income &gt; $50K/yr</div>
        <div className="big grad">{prob == null ? "…" : `${pct.toFixed(0)}%`}</div>
        <div className="probbar" style={{ marginTop: ".7rem" }}><div style={{ width: `${pct}%` }} /></div>
      </div>
    </div>
  );
}
