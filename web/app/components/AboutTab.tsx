"use client";

const ARCHS: [string, string, string][] = [
  ["01 · Feedforward + Autoencoder",
    "The plain multilayer perceptron (MLP): stack linear layers with non-linear activations and it can approximate almost any function. The autoencoder is the same idea folded in half — squeeze the input down to a tiny bottleneck, then rebuild it. What survives the squeeze is the essence of the data, which is why a big rebuild error flags something the model has never seen (an anomaly).",
    "Try the Tabular MLP and Autoencoder tabs."],
  ["02 · Convolutional Networks",
    "Instead of looking at every pixel independently, a CNN slides small filters across the image to detect edges, then textures, then shapes — building understanding layer by layer. We train one from scratch and also fine-tune a ResNet-18 that was pre-trained on millions of images, which is why transfer learning jumps from ~78% to ~96%.",
    "Try the MNIST CNN tab — draw a digit."],
  ["03 · Sequence Models (RNN / LSTM / GRU)",
    "Text and time-series arrive one step at a time, so these models carry a hidden 'memory' from one step to the next. The LSTM adds gates that decide what to remember and what to forget, letting it hold context across a whole sentence.",
    "Try the LSTM Text tab."],
  ["04 · Attention & Transformers",
    "Rather than reading strictly left-to-right, attention lets every word look directly at every other word and weigh what matters. That one idea powers modern language models. We build it from the equations and fine-tune a DistilBERT/RoBERTa.",
    "Try the Transformer (BERT) tab."],
  ["05 · Generative Models (VAE / GAN)",
    "Two ways to invent new images. A VAE learns a smooth 'latent space' you can slide through to morph one digit into another. A GAN pits a generator against a critic in a forgery game until the fakes look real.",
    "Try the VAE and GAN tabs."],
  ["06 · Diffusion Models",
    "Start with a real image and destroy it with noise, step by step, until it's pure static. Then train a network to undo one step of that. To generate, start from static and run the undo many times — a digit emerges. This is the family behind Stable Diffusion and DALL·E.",
    "Try the Diffusion tab."],
  ["07 · Vision Transformer",
    "Take the Transformer built for text and feed it an image — by cutting the image into a grid of patches and treating each patch like a word. Same self-attention, applied to pixels.",
    "Try the Vision Transformer tab (toggle the patch grid)."],
  ["08 · Graph Neural Networks",
    "Some data is a network, not a grid or a sequence — social graphs, molecules, citation links. A GCN classifies each node by repeatedly mixing in information from its neighbors, so a paper's topic is inferred partly from what it cites.",
    "Try the Graph Net (GCN) tab."],
];

export default function AboutTab() {
  return (
    <div className="about" style={{ maxHeight: "min(66vh, 640px)", overflowY: "auto", paddingRight: "1.2rem" }}>
      <p className="note" style={{ fontSize: "1.05rem" }}>
        <strong>Deep Learning Playground</strong> is the interactive companion to an eight-notebook series that builds
        every major deep-learning architecture from first principles in <span className="k">PyTorch</span>. The notebooks
        go deep on the math and the training; this site lets you <em>poke at the finished models</em> and watch them
        think.
      </p>

      <h3><span className="num">✳</span>What is deep learning, in one paragraph?</h3>
      <p>
        A neural network is a long chain of simple math operations with millions of adjustable numbers (weights).
        You show it examples, measure how wrong it is, and nudge every weight a tiny bit in the direction that reduces
        the error — millions of times. That&apos;s <span className="k">training</span>. The art of the field is choosing
        the <em>shape</em> of that chain to match the shape of the data: grids of pixels want convolutions, sequences of
        words want attention, networks of things want graph layers. That&apos;s what the eight notebooks explore.
      </p>

      <h3><span className="num">⚡</span>How does it run in my browser?</h3>
      <p>
        Every model here runs <strong>entirely on your device</strong> — nothing is uploaded, there is no server doing
        the work. Each notebook trains a network in PyTorch and exports it to the open{" "}
        <span className="k">ONNX</span> format (a portable file of the model&apos;s structure + weights). The web app
        loads those files and runs them with <span className="k">ONNX Runtime Web</span> (which executes the math in
        WebAssembly) and <span className="k">Transformers.js</span> for the language model. When you draw a digit, the
        pixels never leave your laptop — the CNN runs right here.
      </p>
      <div className="callout" style={{ margin: "1rem 0" }}>
        <strong>Why are the models small?</strong> The browser versions are deliberately tiny so they load in a second
        and run instantly on a laptop with no GPU. So their accuracy is below a full training run — but the behavior is
        genuine: the CNN really classifies your drawing, the VAE really decodes a latent vector, the diffusion U-Net
        really denoises. For the full-scale results, see the notebooks.
      </div>

      <h3><span className="num">01–08</span>The eight architectures &amp; how they work</h3>
      {ARCHS.map(([t, d, cta], i) => (
        <div key={i} style={{ margin: "1.1rem 0" }}>
          <p style={{ margin: "0 0 .3rem" }}><span className="k">{t}</span></p>
          <p style={{ margin: "0 0 .3rem" }}>{d}</p>
          <p className="note" style={{ margin: 0, fontSize: ".82rem", opacity: 0.8 }}>→ {cta}</p>
        </div>
      ))}

      <h3><span className="num">⌘</span>Stack</h3>
      <p>
        <span className="k">Modeling:</span> PyTorch · torchvision · PyTorch Geometric · Hugging Face Transformers ·
        scikit-learn. <span className="k">Serving:</span> ONNX · ONNX Runtime Web · Transformers.js · Next.js · Vercel.
      </p>

      <p className="note" style={{ margin: "1.4rem 0 .5rem" }}>
        Built by Shivani Bokka. Notebooks &amp; source on{" "}
        <a href="https://github.com/shiva-shivanibokka/Dive-Deeper-into-Deep-Learning" target="_blank" rel="noreferrer">GitHub</a>.
      </p>
    </div>
  );
}
