"use client";

const ARCHS: [string, string][] = [
  ["Feedforward + Autoencoder", "The MLP and the compress-and-rebuild autoencoder — the foundation everything else builds on."],
  ["Convolutional Networks", "A CNN from scratch plus a fine-tuned ResNet-18, with Grad-CAM to see where it looks."],
  ["Sequence Models", "RNN, LSTM, GRU and BiLSTM — reading text one token at a time."],
  ["Attention & Transformers", "Self-attention from the equations up, then a fine-tuned DistilBERT."],
  ["Generative Models", "VAE, DCGAN and WGAN — two different ways to invent new images."],
  ["Diffusion Models", "A DDPM built from scratch: destroy an image with noise, then learn to reverse it."],
  ["Vision Transformer", "An image as a sequence of patches — the Transformer, applied to pixels."],
  ["Graph Neural Networks", "A GCN that classifies nodes by passing messages along a citation graph."],
];

export default function AboutTab() {
  return (
    <div className="about" style={{ maxWidth: 760 }}>
      <p className="note" style={{ fontSize: "1.02rem" }}>
        <strong>Deep Learning Playground</strong> is the interactive companion to an eight-notebook series that builds
        every major deep-learning architecture from first principles in <span className="k">PyTorch</span> — ordered by
        the shape of the data, from flat vectors to grids, sequences, and graphs.
      </p>

      <div className="callout" style={{ margin: "1.4rem 0" }}>
        Every model here runs <strong>entirely in your browser</strong>. The notebooks train each network, export it to
        <span className="k"> ONNX</span>, and the tabs load those weights and run inference on your own machine via
        <span className="k"> ONNX Runtime Web</span> and <span className="k">Transformers.js</span> — no server, no upload,
        nothing leaves your device.
      </div>

      <h3><span className="num">01–08</span>The architectures</h3>
      {ARCHS.map(([t, d], i) => (
        <p key={i}><span className="k">{t}.</span> {d}</p>
      ))}

      <h3><span className="num">✳</span>How the demos work</h3>
      <p>
        The browser models are small, purpose-built versions of the notebook models — trained to be tiny enough for instant
        inference on a laptop. So accuracy is lower than a full training run, but the behavior is real: the CNN genuinely
        classifies your drawing, the VAE genuinely decodes a latent vector, the diffusion U-Net genuinely denoises.
      </p>

      <h3><span className="num">⌘</span>Stack</h3>
      <p>
        <span className="k">Modeling:</span> PyTorch · torchvision · PyTorch Geometric · Hugging Face Transformers · scikit-learn.
        {" "}<span className="k">Serving:</span> ONNX · ONNX Runtime Web · Transformers.js · Next.js · Vercel.
      </p>

      <p className="note" style={{ marginTop: "1.6rem" }}>
        Built by Shivani Bokka. Source & notebooks:{" "}
        <a href="https://github.com/shiva-shivanibokka/Dive-Deeper-into-Deep-Learning" target="_blank" rel="noreferrer">GitHub</a>.
      </p>
    </div>
  );
}
