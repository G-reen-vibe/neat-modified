'use client';

import { useMemo } from 'react';
import type { GenomeDict } from '@/lib/neat-client';

type Props = {
  genome: GenomeDict | null;
  showWeights?: boolean;
  height?: number;
};

const ACTIVATION_COLORS: Record<string, string> = {
  identity: '#94a3b8',
  sigmoid: '#22d3ee',
  tanh: '#a78bfa',
  relu: '#fb923c',
  p_swish: '#34d399',
  uaf: '#f472b6',
};

const KIND_LABELS: Record<string, string> = {
  input: 'Input',
  output: 'Output',
  hidden: 'Hidden',
  bias: 'Bias',
};

/**
 * SVG visualization of a genome's neural network.
 * Inputs on the left, outputs on the right, hidden nodes in the middle.
 * Edge color = sign (blue positive, red negative), thickness = |weight|.
 */
export function GenomeGraph({ genome, showWeights = true, height = 320 }: Props) {
  const layout = useMemo(() => {
    if (!genome) return null;
    const nodes = Object.entries(genome.nodes).map(([id, info]) => ({
      id: parseInt(id, 10),
      kind: info.kind,
      activation: info.activation,
    }));
    const inputs = nodes.filter((n) => n.kind === 'input');
    const outputs = nodes.filter((n) => n.kind === 'output');
    const hidden = nodes.filter((n) => n.kind === 'hidden');
    const bias = nodes.filter((n) => n.kind === 'bias');

    const W = 600;
    const H = height;
    const padX = 60;
    const padY = 40;

    const placeColumn = (arr: typeof nodes, x: number) =>
      arr.map((n, i) => {
        const count = arr.length;
        const y = count === 1 ? H / 2 : padY + (i * (H - 2 * padY)) / (count - 1);
        return { ...n, x, y };
      });

    const placed = [
      ...placeColumn(inputs, padX),
      ...placeColumn(bias, padX),
      ...placeColumn(hidden, W / 2),
      ...placeColumn(outputs, W - padX),
    ];
    const positions = new Map(placed.map((p) => [p.id, p]));

    const edges = Object.entries(genome.conns).map(([innov, c]) => ({
      innov: parseInt(innov, 10),
      from: c.in,
      to: c.out,
      weight: c.w,
      enabled: c.en,
    }));

    return { positions, edges, W, H };
  }, [genome, height]);

  if (!genome || !layout) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-slate-700 bg-slate-900 text-slate-500"
        style={{ height }}
      >
        No genome selected
      </div>
    );
  }

  const maxAbsWeight = Math.max(1, ...layout.edges.map((e) => Math.abs(e.weight)));

  return (
    <svg
      viewBox={`0 0 ${layout.W} ${layout.H}`}
      className="w-full h-auto rounded-lg border border-slate-700 bg-slate-900"
      style={{ maxHeight: height }}
    >
      <defs>
        <marker id="arrow-pos" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#38bdf8" />
        </marker>
        <marker id="arrow-neg" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#f87171" />
        </marker>
      </defs>

      {/* edges */}
      {layout.edges.map((e) => {
        const from = layout.positions.get(e.from);
        const to = layout.positions.get(e.to);
        if (!from || !to) return null;
        const isPos = e.weight >= 0;
        const color = isPos ? '#38bdf8' : '#f87171';
        const opacity = e.enabled ? 0.3 + 0.7 * (Math.abs(e.weight) / maxAbsWeight) : 0.15;
        const strokeW = 0.5 + 3 * (Math.abs(e.weight) / maxAbsWeight);
        return (
          <g key={e.innov}>
            <line
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={color}
              strokeWidth={strokeW}
              strokeOpacity={opacity}
              strokeDasharray={e.enabled ? 'none' : '4 2'}
            />
            {showWeights && Math.abs(e.weight) > 0.1 && (
              <text
                x={(from.x + to.x) / 2}
                y={(from.y + to.y) / 2 - 4}
                fill={color}
                fontSize="9"
                textAnchor="middle"
                className="font-mono"
              >
                {e.weight.toFixed(2)}
              </text>
            )}
          </g>
        );
      })}

      {/* nodes */}
      {Array.from(layout.positions.values()).map((p) => {
        const fill = ACTIVATION_COLORS[p.activation] || '#cbd5e1';
        const stroke = p.kind === 'input' ? '#475569' : p.kind === 'output' ? '#0f766e' : '#334155';
        return (
          <g key={p.id}>
            <circle
              cx={p.x}
              cy={p.y}
              r={14}
              fill={fill}
              stroke={stroke}
              strokeWidth={2}
            />
            <text
              x={p.x}
              y={p.y + 4}
              textAnchor="middle"
              fontSize="10"
              fill="#0f172a"
              fontWeight="bold"
              className="font-mono"
            >
              {p.id}
            </text>
            <text
              x={p.x}
              y={p.y + 28}
              textAnchor="middle"
              fontSize="8"
              fill="#94a3b8"
            >
              {p.activation}
            </text>
          </g>
        );
      })}

      {/* column labels */}
      <text x={60} y={20} textAnchor="middle" fontSize="10" fill="#64748b" className="uppercase tracking-wider">
        Inputs
      </text>
      {Object.values(genome.nodes).some((n) => n.kind === 'hidden') && (
        <text x={layout.W / 2} y={20} textAnchor="middle" fontSize="10" fill="#64748b" className="uppercase tracking-wider">
          Hidden
        </text>
      )}
      <text x={layout.W - 60} y={20} textAnchor="middle" fontSize="10" fill="#64748b" className="uppercase tracking-wider">
        Outputs
      </text>
    </svg>
  );
}
