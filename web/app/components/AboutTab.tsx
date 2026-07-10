"use client";

// Each entry describes how that model actually works *in this project* — the tab,
// the dataset it was trained on, and what you do with it — not deep-learning theory.
const MODELS: [string, string, string][] = [
  ["Tabular MLP", "Notebook 01 · Adult Income",
    "A small feedforward network (linear layers + ReLU) trained to predict whether a person earns over $50K/yr from census features. Move the sliders — age, education, hours, capital gains, marital status, sex — and the exported ONNX model recomputes the probability live on each change."],
  ["Autoencoder", "Notebook 01 · MNIST",
    "The same MLP folded in half: your drawn digit is squeezed through a tiny bottleneck and rebuilt from it. The reconstruction error (shown as a number) is how the notebook flags anomalies — draw a clean digit and it's low; scribble something un-digit-like and it spikes."],
  ["MNIST CNN", "Notebook 02 · MNIST",
    "A convolutional network trained from scratch. It slides small filters over the 28×28 image you draw to pick up edges then shapes, and outputs a probability for each digit 0–9. Your drawing is centered and normalized exactly the way the training data was, then classified pixel-for-pixel in the browser."],
  ["LSTM Text", "Notebook 03 · 20 Newsgroups",
    "A bidirectional LSTM reads your sentence one word at a time, carrying a running memory, and guesses which of six topics it belongs to: hockey, cars, medicine, space, computer graphics, or mideast politics. Type and the topic bars update after a short pause."],
  ["Transformer (BERT)", "Notebook 04 · GoEmotions",
    "A DistilBERT emotion classifier running fully in-browser via Transformers.js. Self-attention lets every word weigh every other word; the model scores your text across 28 fine-grained emotions (Google's GoEmotions taxonomy) and shows the strongest few, so you get a nuanced blend instead of just positive/negative."],
  ["VAE Generator", "Notebook 05 · MNIST",
    "The decoder half of a variational autoencoder. It maps a 2-D latent point to a full digit image. Drag around the latent grid and watch one digit morph smoothly into another — proof the model learned a continuous, meaningful space rather than memorizing examples."],
  ["GAN Generator", "Notebook 05 · MNIST",
    "The generator half of a GAN — the network that was trained against a discriminator in a forgery game until its fakes passed. Feed it a fresh vector of random noise and it paints a digit-like image from scratch."],
  ["Diffusion (DDPM)", "Notebook 06 · MNIST",
    "A denoising U-Net. Generation starts from pure static and runs the model many times, each pass removing a little noise, until a digit emerges. Step through it and watch the image resolve — the same principle behind Stable Diffusion and DALL·E, at MNIST scale."],
  ["Vision Transformer", "Notebook 07 · Fashion-MNIST",
    "The Transformer, applied to images: the picture is cut into a grid of patches, each patch treated like a word, and self-attention does the rest. Toggle the patch grid to see how it carves up the image, then read off its garment prediction."],
  ["Graph Net (GCN)", "Notebook 08 · Cora",
    "A graph convolutional network that classifies research papers by topic using only their citation links — a paper's subject is inferred partly from what it cites. Hover a node to identify a paper; click to light up its citations. It reaches high accuracy from just 140 labeled papers."],
];

export default function AboutTab() {
  return (
    <div className="about" style={{ maxHeight: "min(66vh, 640px)", overflowY: "auto", paddingRight: "1.2rem" }}>
      <p className="note" style={{ fontSize: "1.05rem" }}>
        <strong>Deep Learning Playground</strong> is the interactive companion to an eight-notebook series that builds
        every major deep-learning architecture from scratch in <span className="k">PyTorch</span>. The notebooks go deep
        on the math and the training; this site lets you <em>poke at the finished models</em> and watch them think. Below
        is what each demo actually is and how to use it.
      </p>

      <h3><span className="num">⚡</span>How the models run here</h3>
      <p>
        Every model runs <strong>entirely on your device</strong> — nothing is uploaded, there is no server doing the
        work. Each notebook trains a network in PyTorch and exports it to the open <span className="k">ONNX</span> format
        (a portable file of the model&apos;s structure + weights). The web app loads those files and runs them with{" "}
        <span className="k">ONNX Runtime Web</span> (executing the math in WebAssembly), and{" "}
        <span className="k">Transformers.js</span> for the language model. When you draw a digit or type a sentence, the
        input never leaves your laptop.
      </p>
      <div className="callout" style={{ margin: "1rem 0" }}>
        <strong>Why the demos are small:</strong> the browser versions are deliberately tiny so they load fast and run
        instantly on a laptop with no GPU, so accuracy sits below a full training run. But the behavior is genuine — the
        CNN really classifies your drawing, the VAE really decodes a latent vector, the diffusion U-Net really denoises.
        For the full-scale results, see the notebooks.
      </div>

      <h3><span className="num">◆</span>The models, and what each demo does</h3>
      {MODELS.map(([title, meta, body], i) => (
        <div key={i} style={{ margin: "1.2rem 0" }}>
          <p style={{ margin: "0 0 .15rem" }}>
            <span className="k">{title}</span>
            <span className="note" style={{ fontSize: ".78rem", opacity: 0.75, marginLeft: ".6rem" }}>{meta}</span>
          </p>
          <p style={{ margin: 0 }}>{body}</p>
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
