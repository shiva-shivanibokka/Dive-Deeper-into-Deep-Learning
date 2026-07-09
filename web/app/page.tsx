"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { MODEL_TABS } from "./models";

const load = (p: () => Promise<{ default: React.ComponentType }>) =>
  dynamic(p, { ssr: false, loading: () => <p style={{ color: "var(--muted)" }}>loading model…</p> });

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
      <div className="hero">
        <h1>Deep Learning Playground</h1>
        <p>
          Deep-learning architectures — built from scratch in PyTorch across the companion
          notebooks — running <strong>entirely in your browser</strong> via ONNX Runtime Web. No
          server, no upload, nothing leaves your machine.
        </p>
      </div>

      <div className="tabs" role="tablist" aria-label="Models">
        {MODEL_TABS.map((t) => (
          <button key={t.id} className="tab" role="tab" aria-selected={t.id === active} onClick={() => setActive(t.id)}>
            {t.title}
          </button>
        ))}
      </div>

      <section className="panel" role="tabpanel">
        <div className="panel-head">
          <h2>{tab.title}</h2>
          <span className="chip">Notebook {tab.nb} · {tab.dataset}</span>
        </div>
        <p className="panel-tagline">{tab.tagline}</p>

        {Comp ? (
          <Comp />
        ) : (
          <div className="placeholder">
            <div>
              <strong>Interactive demo coming online</strong>
              This tab will run the trained {tab.title} model live once its weights are exported to ONNX.
            </div>
          </div>
        )}
      </section>

      <p className="footer">
        Built by Shivani Bokka · models trained in PyTorch · served client-side on Vercel
      </p>
    </main>
  );
}
