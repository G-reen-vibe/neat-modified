'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { getNeatClient, type Snapshot, type EpisodeTrace, type GenomeDict } from '@/lib/neat-client';
import { CartPoleCanvas } from '@/components/neat/cartpole-canvas';
import { GenomeGraph } from '@/components/neat/genome-graph';
import { SpeciesPanel } from '@/components/neat/species-panel';
import { StatsPanel } from '@/components/neat/stats-panel';
import { ControlPanel } from '@/components/neat/control-panel';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Brain, Activity, Zap, GitBranch, Wifi, WifiOff } from 'lucide-react';

export default function Home() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [selectedSpeciesId, setSelectedSpeciesId] = useState<number | null>(null);
  const [episodePlaying, setEpisodePlaying] = useState(true);
  const [episodeSpeed, setEpisodeSpeed] = useState(30);
  const clientRef = useRef(getNeatClient());

  useEffect(() => {
    const client = clientRef.current;
    const unsubSnap = client.onSnapshot((snap) => {
      setSnapshot(snap);
      if (typeof snap.running === 'boolean') setRunning(snap.running);
    });
    const unsubStatus = client.onStatus((r) => setRunning(r));
    client.connect();

    // poll connection state
    const interval = setInterval(() => {
      setConnected(client.connected);
    }, 1000);

    return () => {
      unsubSnap();
      unsubStatus();
      clearInterval(interval);
    };
  }, []);

  // Get the latest episode trace from the best genome
  const latestTrace: EpisodeTrace | null = (() => {
    if (!snapshot?.episode_buffer || snapshot.episode_buffer.length === 0) return null;
    return snapshot.episode_buffer[snapshot.episode_buffer.length - 1].trace;
  })();

  // Pick genome to display: best genome, or a genome from the selected species
  const displayGenome: GenomeDict | null = (() => {
    if (!snapshot) return null;
    if (selectedSpeciesId !== null) {
      const fromSpecies = snapshot.genomes.find((g) => g.species_id === selectedSpeciesId);
      if (fromSpecies) return fromSpecies;
    }
    return snapshot.best_genome;
  })();

  const handleSelectSpecies = useCallback((id: number) => {
    setSelectedSpeciesId((cur) => (cur === id ? null : id));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">NEAT-Modified Visualizer</h1>
              <p className="text-xs text-slate-400">CartPole-v1 · GRPO optimizer · Advanced speciation</p>
            </div>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <Badge variant="outline" className={`gap-1 ${connected ? 'border-emerald-500 text-emerald-400' : 'border-rose-500 text-rose-400'}`}>
              {connected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
              {connected ? 'connected' : 'offline'}
            </Badge>
            {running && (
              <Badge variant="outline" className="gap-1 border-cyan-500 text-cyan-400">
                <Activity className="w-3 h-3 animate-pulse" />
                training
              </Badge>
            )}
            <ControlPanel running={running} onRunningChange={setRunning} />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="container mx-auto px-4 py-4 space-y-4">
        {/* Stats row */}
        <StatsPanel history={snapshot?.history ?? []} />

        {/* Visualization row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* CartPole view */}
          <Card className="bg-slate-900 border-slate-700 lg:col-span-1">
            <CardHeader className="pb-2 flex-row items-center justify-between space-y-0">
              <CardTitle className="text-sm text-slate-200 flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-400" />
                Live Agent (Best Genome)
              </CardTitle>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setEpisodePlaying((p) => !p)}
                className="h-7 text-xs bg-slate-800 border-slate-700"
              >
                {episodePlaying ? 'Pause' : 'Play'}
              </Button>
            </CardHeader>
            <CardContent className="pt-2">
              <CartPoleCanvas
                trace={latestTrace}
                playing={episodePlaying}
                speed={episodeSpeed}
              />
              <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
                <span>Speed</span>
                <input
                  type="range"
                  min={5}
                  max={120}
                  value={episodeSpeed}
                  onChange={(e) => setEpisodeSpeed(parseInt(e.target.value, 10))}
                  className="flex-1"
                />
                <span className="font-mono w-10 text-right">{episodeSpeed}fps</span>
              </div>
              {latestTrace && (
                <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                  <div className="text-center p-1.5 rounded bg-slate-800">
                    <div className="text-slate-500">Steps</div>
                    <div className="font-mono text-cyan-400">{latestTrace.steps}</div>
                  </div>
                  <div className="text-center p-1.5 rounded bg-slate-800">
                    <div className="text-slate-500">Reward</div>
                    <div className="font-mono text-emerald-400">{latestTrace.reward.toFixed(0)}</div>
                  </div>
                  <div className="text-center p-1.5 rounded bg-slate-800">
                    <div className="text-slate-500">Solved</div>
                    <div className={`font-mono ${latestTrace.reward >= 475 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {latestTrace.reward >= 475 ? 'YES' : 'NO'}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Genome graph */}
          <Card className="bg-slate-900 border-slate-700 lg:col-span-2">
            <CardHeader className="pb-2 flex-row items-center justify-between space-y-0">
              <CardTitle className="text-sm text-slate-200 flex items-center gap-2">
                <GitBranch className="w-4 h-4 text-violet-400" />
                Genome Network
                {selectedSpeciesId !== null && (
                  <Badge variant="outline" className="ml-2 text-xs">
                    Species #{selectedSpeciesId}
                  </Badge>
                )}
                {selectedSpeciesId === null && snapshot?.best_genome && (
                  <Badge variant="outline" className="ml-2 text-xs border-emerald-500 text-emerald-400">
                    Best
                  </Badge>
                )}
              </CardTitle>
              {displayGenome && (
                <div className="flex gap-3 text-xs text-slate-400">
                  <span>nodes: <span className="font-mono text-slate-200">{Object.keys(displayGenome.nodes).length}</span></span>
                  <span>conns: <span className="font-mono text-slate-200">{Object.keys(displayGenome.conns).length}</span></span>
                  <span>fitness: <span className="font-mono text-emerald-400">{displayGenome.fitness.toFixed(1)}</span></span>
                </div>
              )}
            </CardHeader>
            <CardContent className="pt-2">
              <GenomeGraph genome={displayGenome} height={340} />
            </CardContent>
          </Card>
        </div>

        {/* Species row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-1">
            <SpeciesPanel
              species={snapshot?.species ?? []}
              selectedSpeciesId={selectedSpeciesId}
              onSelectSpecies={handleSelectSpecies}
            />
          </div>
          <Card className="bg-slate-900 border-slate-700 lg:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-slate-200">Top Genomes</CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-700">
                      <th className="text-left py-1.5 px-2">#</th>
                      <th className="text-left py-1.5 px-2">Species</th>
                      <th className="text-right py-1.5 px-2">Fitness</th>
                      <th className="text-right py-1.5 px-2">Nodes</th>
                      <th className="text-right py-1.5 px-2">Conns</th>
                      <th className="text-right py-1.5 px-2">Gen</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(snapshot?.genomes ?? []).slice(0, 20).map((g, i) => (
                      <tr
                        key={i}
                        className="border-b border-slate-800 hover:bg-slate-800/50 cursor-pointer"
                        onClick={() => setSelectedSpeciesId(g.species_id)}
                      >
                        <td className="py-1.5 px-2 text-slate-500 font-mono">{i + 1}</td>
                        <td className="py-1.5 px-2">
                          <Badge variant="outline" className="text-[10px]">#{g.species_id}</Badge>
                        </td>
                        <td className="py-1.5 px-2 text-right font-mono text-emerald-400">{g.fitness.toFixed(1)}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-violet-400">{Object.keys(g.nodes).length}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-cyan-400">{Object.keys(g.conns).length}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-slate-400">{g.generation}</td>
                      </tr>
                    ))}
                    {(!snapshot?.genomes || snapshot.genomes.length === 0) && (
                      <tr>
                        <td colSpan={6} className="text-center text-slate-500 py-8">
                          No genomes yet. Click &ldquo;Start Training&rdquo; to begin.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>

      <footer className="border-t border-slate-800 mt-8">
        <div className="container mx-auto px-4 py-4 text-xs text-slate-500 text-center">
          NEAT-Modified · Python + Next.js · <a href="https://github.com/G-reen-vibe/neat-modified" target="_blank" rel="noreferrer" className="underline hover:text-slate-300">github.com/G-reen-vibe/neat-modified</a>
        </div>
      </footer>
    </div>
  );
}
