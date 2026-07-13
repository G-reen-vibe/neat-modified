'use client';

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, Area, AreaChart,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { HistoryEntry } from '@/lib/neat-client';

type Props = {
  history: HistoryEntry[];
};

export function StatsPanel({ history }: Props) {
  const data = history.map((h) => ({
    gen: h.generation,
    best: h.best_fitness,
    mean: h.mean_fitness,
    species: h.n_species,
    conns: h.avg_conns,
    nodes: h.avg_nodes,
    threshold: h.species_threshold,
  }));

  const latest = history.length > 0 ? history[history.length - 1] : null;
  const bestEver = history.length > 0 ? Math.max(...history.map((h) => h.best_fitness)) : 0;

  return (
    <div className="space-y-3">
      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card className="bg-slate-900 border-slate-700">
          <CardContent className="p-3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Generation</div>
            <div className="text-2xl font-bold text-cyan-400 tabular-nums">
              {latest?.generation ?? 0}
            </div>
          </CardContent>
        </Card>
        <Card className="bg-slate-900 border-slate-700">
          <CardContent className="p-3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Best Fitness</div>
            <div className="text-2xl font-bold text-emerald-400 tabular-nums">
              {bestEver.toFixed(0)}
            </div>
          </CardContent>
        </Card>
        <Card className="bg-slate-900 border-slate-700">
          <CardContent className="p-3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Mean Fitness</div>
            <div className="text-2xl font-bold text-amber-400 tabular-nums">
              {latest?.mean_fitness.toFixed(1) ?? '0.0'}
            </div>
          </CardContent>
        </Card>
        <Card className="bg-slate-900 border-slate-700">
          <CardContent className="p-3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Species</div>
            <div className="text-2xl font-bold text-violet-400 tabular-nums">
              {latest?.n_species ?? 0}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Fitness chart */}
      <Card className="bg-slate-900 border-slate-700">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-slate-200">Fitness Over Generations</CardTitle>
        </CardHeader>
        <CardContent className="pt-2">
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="bestGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="meanGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#fbbf24" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="gen" stroke="#64748b" fontSize={10} />
              <YAxis stroke="#64748b" fontSize={10} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Area type="monotone" dataKey="best" stroke="#34d399" strokeWidth={2} fill="url(#bestGrad)" name="Best" />
              <Area type="monotone" dataKey="mean" stroke="#fbbf24" strokeWidth={2} fill="url(#meanGrad)" name="Mean" />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Topology & species chart */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <Card className="bg-slate-900 border-slate-700">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-slate-200">Network Size</CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="gen" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="conns" stroke="#22d3ee" strokeWidth={2} dot={false} name="Avg Conns" />
                <Line type="monotone" dataKey="nodes" stroke="#a78bfa" strokeWidth={2} dot={false} name="Avg Nodes" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-700">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-slate-200">Species &amp; Threshold</CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="gen" stroke="#64748b" fontSize={10} />
                <YAxis yAxisId="left" stroke="#64748b" fontSize={10} />
                <YAxis yAxisId="right" orientation="right" stroke="#64748b" fontSize={10} domain={[0, 0.5]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line yAxisId="left" type="monotone" dataKey="species" stroke="#fb923c" strokeWidth={2} dot={false} name="Species" />
                <Line yAxisId="right" type="monotone" dataKey="threshold" stroke="#f472b6" strokeWidth={2} dot={false} name="Threshold" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
