"use client";

import { useRef, useEffect, useCallback } from "react";
import { canvasTo28x28 } from "../lib/onnx";

// A 280x280 black canvas you draw white ink on. Calls onResult with a [1,1,28,28]-ready
// Float32Array (length 784, values 0..1) after each stroke, and on clear.
export default function DrawCanvas({ onResult }: { onResult: (data: Float32Array | null) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const dirty = useRef(false);

  const fill = useCallback(() => {
    const c = ref.current!;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.lineWidth = 22;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "white";
  }, []);

  useEffect(() => {
    fill();
  }, [fill]);

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
    const ctx = ref.current!.getContext("2d")!;
    const { x, y } = pos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };
  const move = (e: React.PointerEvent) => {
    if (!drawing.current) return;
    const ctx = ref.current!.getContext("2d")!;
    const { x, y } = pos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
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
    <div style={{ display: "flex", flexDirection: "column", gap: ".6rem", alignItems: "center" }}>
      <canvas
        ref={ref}
        width={280}
        height={280}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerLeave={up}
        style={{ width: 220, height: 220, borderRadius: 10, border: "1px solid var(--border)", touchAction: "none", cursor: "crosshair", background: "#000" }}
      />
      <button className="tab" onClick={clear}>Clear</button>
    </div>
  );
}
