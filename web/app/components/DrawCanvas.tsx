"use client";

import { useRef, useEffect, useCallback } from "react";
import { canvasTo28x28 } from "../lib/onnx";

// A black canvas you draw white ink on. Calls onResult with a length-784 Float32Array
// (values 0..1, MNIST-preprocessed) after each stroke, and null on clear.
export default function DrawCanvas({ onResult, size = 300 }: { onResult: (data: Float32Array | null) => void; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const dirty = useRef(false);

  const fill = useCallback(() => {
    const c = ref.current!;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.lineWidth = 24;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "white";
  }, []);

  useEffect(() => { fill(); }, [fill]);

  const pos = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * ref.current!.width,
      y: ((e.clientY - r.top) / r.height) * ref.current!.height,
    };
  };

  const down = (e: React.PointerEvent) => {
    drawing.current = true;
    dirty.current = true;
    ref.current!.setPointerCapture(e.pointerId);
    const ctx = ref.current!.getContext("2d")!;
    const { x, y } = pos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.1, y + 0.1);
    ctx.stroke();
  };
  const move = (e: React.PointerEvent) => {
    if (!drawing.current) return;
    const ctx = ref.current!.getContext("2d")!;
    const { x, y } = pos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    onResult(canvasTo28x28(ref.current!)); // live prediction as you draw
  };
  const up = () => {
    if (!drawing.current) return;
    drawing.current = false;
    if (dirty.current) onResult(canvasTo28x28(ref.current!));
  };
  const clear = () => {
    fill();
    dirty.current = false;
    onResult(null);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: ".7rem", alignItems: "center" }}>
      <div className="canvas-frame">
        <canvas
          ref={ref}
          className="draw-canvas"
          width={336}
          height={336}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
          style={{ width: size, height: size }}
        />
      </div>
      <button className="btn" onClick={clear}>Clear</button>
    </div>
  );
}
