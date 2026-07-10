import * as ort from "onnxruntime-web";
import { centerGrayIntoFrame } from "./preprocess";

// Single-threaded wasm avoids the COOP/COEP cross-origin-isolation requirement.
// Models are tiny, so one thread is plenty. WASM binaries load from a CDN.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const cache: Record<string, Promise<ort.InferenceSession>> = {};

export function getSession(name: string): Promise<ort.InferenceSession> {
  if (!cache[name]) {
    const p = ort.InferenceSession.create(`/models/${name}`, {
      executionProviders: ["wasm"],
    });
    // Don't cache a rejected load — otherwise one transient failure poisons the
    // tab for the whole session. Drop it so the next call retries.
    p.catch(() => { delete cache[name]; });
    cache[name] = p;
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

// Convert a drawing canvas to a [1,1,28,28] Float32Array in [0,1], using the SAME
// preprocessing MNIST itself uses: crop to the ink's bounding box, scale the longer
// side to 20px, then place it in a 28x28 frame centered on its center of mass.
// Without this, hand-drawn digits are off-center / oversized vs MNIST and every model
// (classifier AND autoencoder) sees an out-of-distribution input.
export function canvasTo28x28(src: HTMLCanvasElement): Float32Array {
  const out = new Float32Array(28 * 28);
  const S = src.width;
  const sctx = src.getContext("2d")!;
  const px = sctx.getImageData(0, 0, S, S).data;

  // bounding box of the ink (red channel; white stroke on black bg)
  let minX = S, minY = S, maxX = -1, maxY = -1;
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      if (px[(y * S + x) * 4] > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  if (maxX < 0) return out; // empty canvas

  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const dw = Math.max(1, Math.round(bw * scale));
  const dh = Math.max(1, Math.round(bh * scale));

  // crop + scale the digit to ~20px on its longer side
  const tmp = document.createElement("canvas");
  tmp.width = dw; tmp.height = dh;
  const tctx = tmp.getContext("2d")!;
  tctx.drawImage(src, minX, minY, bw, bh, 0, 0, dw, dh);
  const dd = tctx.getImageData(0, 0, dw, dh).data;

  // build the scaled grayscale array, then center it by mass into the 28x28 frame
  const scaled = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++)
    for (let x = 0; x < dw; x++)
      scaled[y * dw + x] = dd[(y * dw + x) * 4] / 255;
  return centerGrayIntoFrame(scaled, dw, dh);
}
