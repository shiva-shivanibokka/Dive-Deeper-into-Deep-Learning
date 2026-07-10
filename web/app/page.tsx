"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { MODEL_TABS } from "./models";

const load = (p: () => Promise<{ default: React.ComponentType }>) =>
  dynamic(p, { ssr: false, loading: () => <p className="note">loading demo…</p> });

const TAB_COMPONENTS: Record<string, React.ComponentType> = {
  mlp: load(() => import("./components/MlpTab")),
  lstm: load(() => import("./components/LstmTab")),
  transformer: load(() => import("./components/BertTab")),
  vae: load(() => import("./components/VaeTab")),
  gan: load(() => import("./components/GanTab")),
  cnn: load(() => import("./components/CnnTab")),
  autoencoder: load(() => import("./components/AutoencoderTab")),
  diffusion: load(() => import("./components/DiffusionTab")),
  vit: load(() => import("./components/VitTab")),
  gnn: load(() => import("./components/GnnTab")),
};

export default function Home() {
  const [active, setActive] = useState(MODEL_TABS[0].id);
  const tab = MODEL_TABS.find((t) => t.id === active)!;
  const Comp = TAB_COMPONENTS[tab.id];

  return (
    <main className="wrap">
      <header className="hero">
        <h1>Deep Learning Playground</h1>
        <p>
          Ten deep-learning architectures — from a humble MLP to Transformers, diffusion models and graph nets —
          built from scratch in PyTorch across the companion notebooks and running{" "}
          <strong>entirely in your browser</strong>. Every prediction runs on your own machine via ONNX Runtime Web
          and Transformers.js. Nothing is uploaded.
        </p>
        <span className="live">
          <b>●</b> live · no server · nothing leaves your machine
        </span>
      </header>

      <nav className="tabs" role="tablist" aria-label="Models">
        {MODEL_TABS.map((t) => (
          <button key={t.id} className="tab" role="tab" aria-selected={t.id === active} onClick={() => setActive(t.id)}>
            {t.title}
          </button>
        ))}
      </nav>

      <section className="panel" role="tabpanel">
        <div className="panel-head">
          <h2>{tab.title}</h2>
          <span className="chip">Notebook {tab.nb} · {tab.dataset}</span>
        </div>
        <p className="panel-tagline">{tab.tagline}</p>
        {Comp ? <Comp /> : null}
      </section>

      <p className="footer">
        Built by Shivani Bokka · trained in PyTorch · served client-side on Vercel
      </p>
    </main>
  );
}
