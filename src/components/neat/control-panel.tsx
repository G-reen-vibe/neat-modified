'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Play, Square, Loader2, Settings } from 'lucide-react';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from '@/components/ui/dialog';
import { getNeatClient } from '@/lib/neat-client';

type Props = {
  running: boolean;
  onRunningChange: (running: boolean) => void;
};

export function ControlPanel({ running, onRunningChange }: Props) {
  const [speed, setSpeed] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [starting, setStarting] = useState(false);
  const [config, setConfig] = useState({
    pop_size: 80,
    generations: 100,
    n_avg: 1,
    init_neurons: 1,
    init_mult: 2.0,
    weight_prob: 0.8,
    weight_pct: 0.3,
    weight_std: 0.05,
    connection_prob: 0.1,
    neuron_prob: 0.05,
    pruning_prob: 0.02,
    target_species: 5,
    threshold: 0.25,
    elitism: 3,
    cull_pct: 0.6,
    optimizer_enabled: false,
    opt_lr: 0.02,
    opt_method: 'adam',
    seed: 0,
  });

  const client = getNeatClient();

  const handleStart = async () => {
    setStarting(true);
    try {
      await client.start(config);
      onRunningChange(true);
    } finally {
      setStarting(false);
    }
  };

  const handleStop = async () => {
    await client.stop();
    onRunningChange(false);
  };

  const handleSpeedChange = async (val: number[]) => {
    const v = val[0];
    setSpeed(v);
    await client.setDelay(v === 0 ? 0 : 1.0 / v);
  };

  return (
    <div className="flex items-center gap-3 flex-wrap">
      {!running ? (
        <Button
          onClick={handleStart}
          disabled={starting}
          size="sm"
          className="bg-emerald-600 hover:bg-emerald-700 text-white"
        >
          {starting ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Play className="w-4 h-4 mr-1" />}
          Start Training
        </Button>
      ) : (
        <Button
          onClick={handleStop}
          size="sm"
          variant="destructive"
        >
          <Square className="w-4 h-4 mr-1" />
          Stop
        </Button>
      )}

      <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-800 border border-slate-700">
        <Label className="text-xs text-slate-400">Speed</Label>
        <Slider
          value={[speed]}
          onValueChange={handleSpeedChange}
          min={0}
          max={20}
          step={1}
          className="w-32"
        />
        <span className="text-xs text-slate-300 font-mono w-12">
          {speed === 0 ? 'max' : `${speed}/s`}
        </span>
      </div>

      <Dialog open={showSettings} onOpenChange={setShowSettings}>
        <DialogTrigger asChild>
          <Button variant="outline" size="sm" className="bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700">
            <Settings className="w-4 h-4 mr-1" />
            Settings
          </Button>
        </DialogTrigger>
        <DialogContent className="bg-slate-900 border-slate-700 text-slate-200 max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Training Configuration</DialogTitle>
          </DialogHeader>
          <div className="grid grid-cols-2 gap-3 py-2">
            <Field label="Population Size" value={config.pop_size} onChange={(v) => setConfig({ ...config, pop_size: v })} min={10} max={500} step={10} />
            <Field label="Generations" value={config.generations} onChange={(v) => setConfig({ ...config, generations: v })} min={5} max={500} step={5} />
            <Field label="Episodes/Genome" value={config.n_avg} onChange={(v) => setConfig({ ...config, n_avg: v })} min={1} max={10} step={1} />
            <Field label="Init Neurons" value={config.init_neurons} onChange={(v) => setConfig({ ...config, init_neurons: v })} min={0} max={5} step={1} />
            <Field label="Init Conn Multiplier" value={config.init_mult} onChange={(v) => setConfig({ ...config, init_mult: v })} min={0.5} max={5} step={0.5} />
            <Field label="Weight Mutation Prob" value={config.weight_prob} onChange={(v) => setConfig({ ...config, weight_prob: v })} min={0} max={1} step={0.05} />
            <Field label="Weight Mutation %" value={config.weight_pct} onChange={(v) => setConfig({ ...config, weight_pct: v })} min={0} max={1} step={0.05} />
            <Field label="Weight Std" value={config.weight_std} onChange={(v) => setConfig({ ...config, weight_std: v })} min={0.01} max={0.5} step={0.01} />
            <Field label="Connection Prob" value={config.connection_prob} onChange={(v) => setConfig({ ...config, connection_prob: v })} min={0} max={1} step={0.05} />
            <Field label="Neuron Prob" value={config.neuron_prob} onChange={(v) => setConfig({ ...config, neuron_prob: v })} min={0} max={1} step={0.05} />
            <Field label="Pruning Prob" value={config.pruning_prob} onChange={(v) => setConfig({ ...config, pruning_prob: v })} min={0} max={0.5} step={0.01} />
            <Field label="Target Species" value={config.target_species} onChange={(v) => setConfig({ ...config, target_species: v })} min={1} max={20} step={1} />
            <Field label="Species Threshold" value={config.threshold} onChange={(v) => setConfig({ ...config, threshold: v })} min={0.05} max={0.5} step={0.025} />
            <Field label="Elitism" value={config.elitism} onChange={(v) => setConfig({ ...config, elitism: v })} min={1} max={10} step={1} />
            <Field label="Cull %" value={config.cull_pct} onChange={(v) => setConfig({ ...config, cull_pct: v })} min={0.1} max={0.9} step={0.1} />
            <Field label="Seed" value={config.seed} onChange={(v) => setConfig({ ...config, seed: v })} min={0} max={1000} step={1} />
          </div>
          <div className="flex items-center gap-3 py-2 border-t border-slate-700">
            <Switch
              checked={config.optimizer_enabled}
              onCheckedChange={(c) => setConfig({ ...config, optimizer_enabled: c })}
            />
            <Label className="text-sm">Enable GRPO Optimizer</Label>
            {config.optimizer_enabled && (
              <>
                <Field label="LR" value={config.opt_lr} onChange={(v) => setConfig({ ...config, opt_lr: v })} min={0.001} max={0.5} step={0.005} />
                <select
                  value={config.opt_method}
                  onChange={(e) => setConfig({ ...config, opt_method: e.target.value })}
                  className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs"
                >
                  <option value="sgd">SGD</option>
                  <option value="momentum">Momentum</option>
                  <option value="rmsprop">RMSProp</option>
                  <option value="adam">Adam</option>
                </select>
              </>
            )}
          </div>
          <div className="text-xs text-slate-500 py-2">
            Note: changes apply only when starting a new training run.
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function Field({ label, value, onChange, min, max, step }: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <div className="space-y-1">
      <Label className="text-xs text-slate-400">{label}</Label>
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="bg-slate-800 border-slate-700 text-slate-200 h-8 text-xs"
      />
    </div>
  );
}
