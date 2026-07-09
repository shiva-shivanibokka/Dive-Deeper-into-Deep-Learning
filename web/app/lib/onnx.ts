import * as ort from "onnxruntime-web";

// Single-threaded wasm avoids the COOP/COEP cross-origin-isolation requirement.
// Models are tiny, so one thread is plenty. WASM binaries load from a CDN.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const cache: Record<string, Promise<ort.InferenceSession>> = {};

export function getSession(name: string): Promise<ort.InferenceSession> {
  if (!cache[name]) {
    cache[name] = ort.InferenceSession.create(`/models/${name}`, {
      executionProviders: ["wasm"],
    });
  }
  return cache[name];
}

export { ort };

// Render a single-channel [h*w] Float32Array into a display canvas, nearest-neighbor upscaled.
// `norm` maps a raw model value to 0..1 (e.g. tanh outputs use (v+1)/2).
export function renderGray(
  canvas: HTMLCanvasElement,
  data: Float32Array | ort.Tensor["data"],
  w: number,
  h: number,
  norm: (v: number) => number = (v) => v,
) {
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  const octx = off.getContext("2d")!;
  const img = octx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(1, norm(Number((data as ArrayLike<number>)[i]))));
    const px = Math.round(v * 255);
    img.data[i * 4] = px;
    img.data[i * 4 + 1] = px;
    img.data[i * 4 + 2] = px;
    img.data[i * 4 + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

// Downscale a drawing canvas to 28x28 and return a [1,1,28,28] Float32Array in [0,1] (white ink on black).
export function canvasTo28x28(src: HTMLCanvasElement): Float32Array {
  const off = document.createElement("canvas");
  off.width = 28;
  off.height = 28;
  const octx = off.getContext("2d")!;
  octx.drawImage(src, 0, 0, 28, 28);
  const { data } = octx.getImageData(0, 0, 28, 28);
  const out = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    // src is white stroke on black bg; use the red channel, normalize to [0,1]
    out[i] = data[i * 4] / 255;
  }
  return out;
}
