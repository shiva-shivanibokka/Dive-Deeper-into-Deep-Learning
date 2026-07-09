// Single source of truth for the demo tabs. Each entry maps to one notebook's model.
// A tab renders its interactive component if page.tsx's TAB_COMPONENTS has its id;
// otherwise it shows the "coming online" placeholder.
export type ModelTab = {
  id: string;
  nb: string;
  title: string;
  tagline: string;
  dataset: string;
};

export const MODEL_TABS: ModelTab[] = [
  { id: "mlp",         nb: "01", title: "Tabular MLP",        tagline: "Predict income from a form of features.",       dataset: "Adult Income" },
  { id: "autoencoder", nb: "01", title: "Autoencoder",        tagline: "Draw a digit; watch it compress & rebuild.",    dataset: "MNIST" },
  { id: "cnn",         nb: "02", title: "MNIST CNN",          tagline: "Draw a digit; a CNN classifies it live.",       dataset: "MNIST" },
  { id: "lstm",        nb: "03", title: "LSTM Text",          tagline: "Type text; an LSTM guesses the topic.",         dataset: "20 Newsgroups" },
  { id: "transformer", nb: "04", title: "Transformer (BERT)", tagline: "In-browser DistilBERT sentiment analysis.",     dataset: "text" },
  { id: "vae",         nb: "05", title: "VAE Generator",      tagline: "Move latent sliders to generate digits live.",  dataset: "MNIST" },
  { id: "gan",         nb: "05", title: "GAN Generator",      tagline: "Sample random noise into fresh digits.",        dataset: "MNIST" },
  { id: "diffusion",   nb: "06", title: "Diffusion (DDPM)",   tagline: "Watch pure noise denoise into a digit.",        dataset: "MNIST" },
  { id: "vit",         nb: "07", title: "Vision Transformer", tagline: "Patch-based Fashion-MNIST classification.",     dataset: "Fashion-MNIST" },
  { id: "gnn",         nb: "08", title: "Graph Net (GCN)",    tagline: "Explore a GCN's predictions on a citation graph.", dataset: "Cora" },
];
