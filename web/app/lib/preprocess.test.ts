import { describe, it, expect } from "vitest";
import { softmax, tokenize, encodeTokens, centerGrayIntoFrame } from "./preprocess";

describe("softmax", () => {
  it("sums to 1, is non-negative, preserves the argmax", () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p.indexOf(Math.max(...p))).toBe(2);
    p.forEach((v) => expect(v).toBeGreaterThanOrEqual(0));
  });
  it("stays finite for large logits (numerical stability)", () => {
    const p = softmax([1000, 1001, 999]);
    expect(p.every(Number.isFinite)).toBe(true);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });
});

describe("tokenize", () => {
  it("lowercases and splits on [a-z] runs, dropping digits and punctuation", () => {
    expect(tokenize("Hello, WORLD! 42 don't")).toEqual(["hello", "world", "don", "t"]);
  });
  it("returns [] when there are no letters", () => {
    expect(tokenize("  123 !!! ")).toEqual([]);
  });
});

describe("encodeTokens", () => {
  const vocab = { the: 2, cat: 3 };
  it("maps unknown words to 1 (<unk>) and right-pads with 0 (<pad>)", () => {
    expect(encodeTokens("the dog", vocab, 5)).toEqual([2, 1, 0, 0, 0]);
  });
  it("truncates to maxlen", () => {
    expect(encodeTokens("the cat the cat the cat", vocab, 3)).toEqual([2, 3, 2]);
  });
});

describe("centerGrayIntoFrame (MNIST centering)", () => {
  it("moves an off-center blob's center of mass to the frame center", () => {
    const w = 6, h = 6;
    const g = new Float32Array(w * h); // 2x2 white block in the top-left corner
    for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) g[y * w + x] = 1;
    const out = centerGrayIntoFrame(g, w, h, 28, 14);
    let sum = 0, cx = 0, cy = 0;
    for (let y = 0; y < 28; y++)
      for (let x = 0; x < 28; x++) { const v = out[y * 28 + x]; sum += v; cx += v * x; cy += v * y; }
    expect(sum).toBe(4);
    expect(cx / sum).toBeGreaterThanOrEqual(13);
    expect(cx / sum).toBeLessThanOrEqual(15);
    expect(cy / sum).toBeGreaterThanOrEqual(13);
    expect(cy / sum).toBeLessThanOrEqual(15);
  });
  it("returns a blank 28x28 frame for empty input", () => {
    const out = centerGrayIntoFrame(new Float32Array(16), 4, 4);
    expect(out.length).toBe(784);
    expect(out.every((v) => v === 0)).toBe(true);
  });
});
