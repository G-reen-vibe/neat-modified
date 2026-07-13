'use client';

import { useEffect, useRef } from 'react';
import type { EpisodeTrace } from '@/lib/neat-client';

type Props = {
  trace: EpisodeTrace | null;
  playing: boolean;
  speed: number;
  onStep?: (step: number) => void;
};

/**
 * Renders a CartPole episode trace step-by-step on a canvas.
 *
 * CartPole-v1 state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
 */
export function CartPoleCanvas({ trace, playing, speed, onStep }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stepRef = useRef(0);
  const lastTimeRef = useRef(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    stepRef.current = 0;
    lastTimeRef.current = 0;
  }, [trace]);

  useEffect(() => {
    if (!playing || !trace) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const trackY = H * 0.75;
    const cartW = 60;
    const cartH = 30;
    const poleLen = 100;
    const worldScale = 80;

    const draw = (t: number) => {
      if (!trace) return;
      if (lastTimeRef.current === 0) lastTimeRef.current = t;
      const dt = t - lastTimeRef.current;
      lastTimeRef.current = t;
      const stepsToAdvance = Math.max(0, Math.floor((dt / 1000) * speed));
      if (stepsToAdvance > 0) {
        stepRef.current = Math.min(stepRef.current + stepsToAdvance, trace.obs.length - 1);
        onStep?.(stepRef.current);
      }
      const idx = stepRef.current;
      const obs = trace.obs[idx];
      if (!obs) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }
      const [cartPos, , poleAngle] = obs;

      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let x = 0; x < W; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      }
      for (let y = 0; y < H; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      }

      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, trackY);
      ctx.lineTo(W, trackY);
      ctx.stroke();

      const cx = W / 2 + cartPos * worldScale;
      const cy = trackY - cartH / 2;
      ctx.fillStyle = '#22d3ee';
      ctx.fillRect(cx - cartW / 2, cy - cartH / 2, cartW, cartH);
      ctx.strokeStyle = '#0891b2';
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - cartW / 2, cy - cartH / 2, cartW, cartH);
      ctx.fillStyle = '#0e7490';
      ctx.beginPath();
      ctx.arc(cx - cartW / 4, trackY, 6, 0, Math.PI * 2);
      ctx.arc(cx + cartW / 4, trackY, 6, 0, Math.PI * 2);
      ctx.fill();

      const poleEndX = cx + Math.sin(poleAngle) * poleLen;
      const poleEndY = cy - Math.cos(poleAngle) * poleLen;
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 6;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(poleEndX, poleEndY);
      ctx.stroke();
      ctx.fillStyle = '#fbbf24';
      ctx.beginPath();
      ctx.arc(poleEndX, poleEndY, 8, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#334155';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(W / 2, 0);
      ctx.lineTo(W / 2, H);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#e2e8f0';
      ctx.font = '13px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`step ${idx + 1}/${trace.obs.length}`, 12, 20);
      ctx.fillText(`reward ${trace.reward.toFixed(0)}`, 12, 38);
      ctx.fillText(`pos ${cartPos.toFixed(2)}`, 12, 56);
      ctx.fillText(`angle ${poleAngle.toFixed(3)}`, 12, 74);
      const action = trace.actions[idx];
      ctx.textAlign = 'right';
      ctx.fillStyle = action === 0 ? '#a5f3fc' : '#fcd34d';
      ctx.fillText(`action: ${action === 0 ? 'LEFT' : 'RIGHT'}`, W - 12, 20);

      if (stepRef.current >= trace.obs.length - 1) {
        stepRef.current = 0;
        lastTimeRef.current = 0;
      }
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [playing, trace, speed, onStep]);

  return (
    <canvas
      ref={canvasRef}
      width={640}
      height={360}
      className="w-full h-auto rounded-lg border border-slate-700 bg-slate-900"
    />
  );
}
