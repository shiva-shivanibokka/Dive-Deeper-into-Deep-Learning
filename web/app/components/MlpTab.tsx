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

  useEffect(() => {
    fetch("/models/tabular_meta.json").then((r) => r.json()).then((m: Meta) => {
      setMeta(m);
      setVals({ ...m.defaults });
    });
  }, []);

  const predict = useCallback(async (m: Meta, v: Record<string, number>) => {
    // Build the feature vector in the exact training order; standardize the numerics.
    const vec = m.features.map((f) => {
      const ni = m.numeric.indexOf(f);
      if (ni >= 0) return (v[f] - m.mean[ni]) / m.std[ni];
      return v[f]; // binary flags already 0/1
    });
    const sess = await getSession("tabular_mlp.onnx");
    const out = await sess.run({ features: new ort.Tensor("float32", Float32Array.from(vec), [1, vec.length]) });
    const logit = Number((out.logit.data as Float32Array)[0]);
    setProb(1 / (1 + Math.exp(-logit)));
  }, []);

  useEffect(() => {
    if (meta && Object.keys(vals).length) predict(meta, vals);
  }, [meta, vals, predict]);

  if (!meta) return <p style={{ color: "var(--muted)" }}>loading model…</p>;
  const set = (k: string, n: number) => setVals((p) => ({ ...p, [k]: n }));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: "2rem", alignItems: "center" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        {meta.numeric.map((f) => (
          <label key={f} style={{ fontSize: ".85rem" }}>
            <span style={{ color: "var(--muted)" }}>{LABELS[f] ?? f}: </span>
            <strong>{vals[f]}</strong>
            <input type="range" min={meta.ranges[f][0]} max={meta.ranges[f][1]} value={vals[f] ?? 0}
              onChange={(e) => set(f, +e.target.value)} style={{ width: "100%" }} />
          </label>
        ))}
        <div style={{ display: "flex", gap: "1.5rem", marginTop: ".3rem" }}>
          <Toggle label="Married" on={vals.is_married === 1} onClick={() => set("is_married", vals.is_married === 1 ? 0 : 1)} />
          <Toggle label="Male" on={vals.sex_is_male === 1} onClick={() => set("sex_is_male", vals.sex_is_male === 1 ? 0 : 1)} />
        </div>
      </div>

      <div style={{ textAlign: "center" }}>
        <div style={{ color: "var(--muted)", fontSize: ".8rem" }}>Predicted probability of</div>
        <div style={{ color: "var(--muted)", fontSize: ".8rem", marginBottom: ".5rem" }}>income &gt; $50K / yr</div>
        <div style={{ fontSize: "3rem", fontWeight: 800, color: prob != null && prob > 0.5 ? "var(--accent)" : "var(--accent-2)" }}>
          {prob == null ? "…" : `${(prob * 100).toFixed(0)}%`}
        </div>
        <div style={{ height: 10, background: "var(--panel-2)", borderRadius: 6, marginTop: ".6rem" }}>
          <div style={{ width: `${(prob ?? 0) * 100}%`, height: "100%", background: "var(--accent)", borderRadius: 6, transition: "width .15s" }} />
        </div>
      </div>
    </div>
  );
}

function Toggle({ label, on, onClick }: { label: string; on: boolean; onClick: () => void }) {
  return (
    <button className="tab" aria-selected={on} onClick={onClick} style={{ minWidth: 90 }}>
      {label}: {on ? "Yes" : "No"}
    </button>
  );
}
