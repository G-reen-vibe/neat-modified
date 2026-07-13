'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { SpeciesInfo } from '@/lib/neat-client';

type Props = {
  species: SpeciesInfo[];
  selectedSpeciesId: number | null;
  onSelectSpecies: (id: number) => void;
};

const SPECIES_COLORS = [
  '#22d3ee', '#a78bfa', '#fb923c', '#34d399', '#f472b6',
  '#facc15', '#60a5fa', '#fb7185', '#2dd4bf', '#c084fc',
  '#fdba74', '#86efac', '#f9a8d4', '#fde68a', '#93c5fd',
];

export function SpeciesPanel({ species, selectedSpeciesId, onSelectSpecies }: Props) {
  const maxSize = Math.max(1, ...species.map((s) => s.size));
  const totalMembers = species.reduce((sum, s) => sum + s.size, 0);

  return (
    <Card className="bg-slate-900 border-slate-700">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm text-slate-200 flex items-center justify-between">
          <span>Species ({species.length})</span>
          <span className="text-xs text-slate-400 font-normal">{totalMembers} genomes</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[280px]">
          <div className="px-3 pb-3 space-y-1.5">
            {species.length === 0 && (
              <div className="text-center text-slate-500 py-8 text-sm">No species yet</div>
            )}
            {species.map((sp, idx) => {
              const color = SPECIES_COLORS[idx % SPECIES_COLORS.length];
              const isSelected = sp.id === selectedSpeciesId;
              const widthPct = (sp.size / maxSize) * 100;
              return (
                <button
                  key={sp.id}
                  onClick={() => onSelectSpecies(sp.id)}
                  className={`w-full text-left p-2 rounded-md transition-colors ${
                    isSelected ? 'bg-slate-700/70' : 'hover:bg-slate-800'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs mb-1">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <span className="text-slate-200 font-mono">#{sp.id}</span>
                      {sp.staleness > 5 && (
                        <Badge variant="outline" className="text-[10px] h-4 px-1 border-amber-500 text-amber-500">
                          stale {sp.staleness}
                        </Badge>
                      )}
                    </div>
                    <span className="text-slate-400">{sp.size}</span>
                  </div>
                  <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{ width: `${widthPct}%`, backgroundColor: color }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                    <span>best {sp.best_fitness.toFixed(0)}</span>
                    <span>avg {sp.avg_fitness.toFixed(1)}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
