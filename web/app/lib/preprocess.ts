// Pure, DOM-free preprocessing helpers shared across tabs and covered by unit tests.
// Keeping these in one place removes the softmax/tokenizer duplication the audit
// flagged and lets CI guard the export<->inference contract.

/** Numerically stable softmax over a logit array. */
export function softmax(a: number[]): number[] {
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
}

/** Same tokenizer the LSTM was trained with: lowercase, split on runs of [a-z]. */
export function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z]+/g) ?? [];
}

/**
 * Encode text to a fixed-length token-id sequence exactly as export_lstm.py does:
 * map each token via word2idx (1 = <unk>), truncate to maxlen, right-pad with 0 (<pad>).
 */
export function encodeTokens(
  text: string,
  word2idx: Record<string, number>,
  maxlen: number,
): number[] {
  const ids = tokenize(text).slice(0, maxlen).map((w) => word2idx[w] ?? 1);
  while (ids.length < maxlen) ids.push(0);
  return ids;
}

/**
 * Place a small grayscale image (values in [0,1]) into a `frame`x`frame` output so its
 * center of mass lands on (`target`,`target`) — the MNIST centering the CNN/autoencoder
 * were trained with. Returns a zeroed frame if the input is blank.
 */
export function centerGrayIntoFrame(
  gray: Float32Array | number[],
  w: number,
  h: number,
  frame = 28,
  target = 14,
): Float32Array {
  const out = new Float32Array(frame * frame);
  let sum = 0, cx = 0, cy = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = gray[y * w + x];
      sum += v; cx += v * x; cy += v * y;
    }
  if (sum === 0) return out;
  cx /= sum; cy /= sum;
  const ox = Math.round(target - cx), oy = Math.round(target - cy);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const tx = x + ox, ty = y + oy;
      if (tx >= 0 && tx < frame && ty >= 0 && ty < frame) out[ty * frame + tx] = gray[y * w + x];
    }
  return out;
}
