"""
Population: drives one generation of evolution.

Per spec's Generation Policy:
    For each species (or the entire population if no species):
        - Generate X% of the group asexually+mutation
        - Generate Y% of the group with crossover
    Then perform W interspecies crossovers and hold Z genomes (elitism).
    Cull some proportion of each species/population such that they cannot
    generate any children.
    Use roulette wheel on rewards for parent selection; for crossover, spin
    twice and avoid reselecting the same genome.

Order of operations per generation:
    1. Evaluate all genomes (caller's responsibility - via .evaluate())
    2. Speciate
    3. Compute partial gradients & revert weight mutations (optimizer)
    4. Cull bottom C% of each species
    5. Apply optimizer step (per-species)
    6. Reproduce (asexual + crossover + interspecies + elitism)
    7. Apply mutation policy to all children
    8. Advance generation counter

Notes:
    - The spec says "Mutations are always applied at the very end, after
      crossover, speciation, and evaluation."  We follow this.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Callable, Any
import numpy as np

from .config import Config
from .genome import Genome
from .indexing import GlobalIndex
from .speciation import Speciator, Species
from .crossover import crossover, asexual
from .mutations import apply_mutation_policy
from .optimizer import Optimizer
from .similarity import similarity
from .initialization import initialize_genome


class Population:
    """A population of genomes evolving under NEAT."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.index = GlobalIndex(cfg.n_inputs, cfg.n_outputs, cfg.bias_enabled)
        self.speciator = Speciator(cfg)
        self.optimizer = Optimizer(cfg)
        self.genomes: List[Genome] = []
        self.generation: int = 0
        self._next_gid: int = 0
        self.rng = np.random.default_rng(cfg.seed)
        self.history: List[Dict[str, Any]] = []
        # Initialize population
        self._initialize()

    # ------------------------------------------------------------------
    def _new_genome(self) -> Genome:
        g = Genome(self.cfg, self.index)
        g.id = self._next_gid
        self._next_gid += 1
        return g

    def _initialize(self) -> None:
        """Build the initial population."""
        self.genomes = []
        for _ in range(self.cfg.generation.pop_size):
            g = self._new_genome()
            # Use the initialization module (partitioned conn + neuron mutations)
            g2 = initialize_genome(self.cfg, self.index, self.rng)
            g2.id = g.id
            self.genomes.append(g2)

    # ------------------------------------------------------------------
    def evaluate(self, eval_fn: Callable[[Genome], float]) -> None:
        """Evaluate fitness for every genome."""
        for g in self.genomes:
            g.fitness = float(eval_fn(g))

    # ------------------------------------------------------------------
    def step(self) -> Dict[str, Any]:
        """Run one full generation step.  Returns a stats dict."""
        # 1. Evaluate is assumed done by caller before .step()
        # 2. Speciate
        self.speciator.speciate(self.genomes)
        # 3. Optimizer: compute partial gradients & revert weight mutations
        #    (per-species; only if enabled)
        if self.cfg.optimizer.enabled:
            for s in self.speciator.species.values():
                members = [g for g in self.genomes if g.id in s.members]
                self.optimizer.step(members, self.genomes)
        else:
            # Still need to consume the deltas so they don't accumulate
            for g in self.genomes:
                g.consume_weight_deltas()
        # 4. Cull bottom C% of each species (they cannot reproduce)
        survivors_per_species = self._cull()
        # 5. Reproduce
        new_genomes, elite_ids = self._reproduce(survivors_per_species)
        # 6. Mutations applied at the very end (per spec); elites are exempt
        from .mutations import repair_genome
        for g in new_genomes:
            if g.id in elite_ids:
                continue  # elitism: don't mutate
            apply_mutation_policy(g, self.cfg, self.rng)
            # Repair: ensure every output has at least one incoming connection
            repair_genome(g, self.rng)
        # Replace population
        self.genomes = new_genomes
        self.generation += 1
        self.speciator.advance_generation()
        # 7. Stats
        stats = self._compute_stats()
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------------
    def _cull(self) -> Dict[int, List[Genome]]:
        """Cull the bottom cull_pct of each species.  Returns survivors per species."""
        cfg = self.cfg
        cull_pct = cfg.generation.cull_pct
        survivors: Dict[int, List[Genome]] = {}
        for sid, s in self.speciator.species.items():
            members = [g for g in self.genomes if g.id in s.members]
            members.sort(key=lambda g: g.fitness, reverse=True)
            n_keep = max(1, int(round(len(members) * (1 - cull_pct))))
            survivors[sid] = members[:n_keep]
        return survivors

    # ------------------------------------------------------------------
    def _roulette(self, candidates: List[Genome], n: int,
                  exclude: Optional[Genome] = None) -> List[Genome]:
        """Standard roulette wheel selection.  Never returns the same genome twice
        in a single call (for crossover)."""
        if not candidates:
            return []
        fitnesses = np.array([max(g.fitness, 1e-6) for g in candidates])
        probs = fitnesses / fitnesses.sum()
        chosen: List[Genome] = []
        avail_idx = list(range(len(candidates)))
        for _ in range(n):
            if not avail_idx:
                break
            p = probs[avail_idx]
            p = p / p.sum()
            pick = self.rng.choice(avail_idx, p=p)
            avail_idx.remove(pick)
            chosen.append(candidates[pick])
        return chosen

    # ------------------------------------------------------------------
    def _reproduce(self, survivors_per_species: Dict[int, List[Genome]]) -> Tuple[List[Genome], set]:
        """Produce next generation per the spec's generation policy.

        Returns (new_genomes, elite_ids) where elite_ids is the set of new
        genome ids that were carried over as elites (and should not be mutated).
        """
        cfg = self.cfg
        pop_size = cfg.generation.pop_size
        n_asexual = int(round(pop_size * cfg.generation.asexual_pct))
        n_crossover = int(round(pop_size * cfg.generation.crossover_pct))
        n_interspecies = cfg.generation.interspecies
        n_elite = cfg.generation.elitism

        # Reserve slots for interspecies + elitism
        n_asexual = max(0, n_asexual)
        n_crossover = max(0, min(n_crossover, pop_size - n_asexual))
        # Adjust if total exceeds pop_size
        total = n_asexual + n_crossover + n_interspecies + n_elite
        if total > pop_size:
            # Trim asexual first, then crossover
            overflow = total - pop_size
            for _ in range(overflow):
                if n_asexual > 0:
                    n_asexual -= 1
                elif n_crossover > 0:
                    n_crossover -= 1
                elif n_interspecies > 0:
                    n_interspecies -= 1
                elif n_elite > 0:
                    n_elite -= 1

        new_genomes: List[Genome] = []

        # --- Elitism: carry over the best `n_elite` genomes (unchanged) ---
        # Per spec, "hold Z genomes (elitism)"; we interpret this as Z total
        # (the absolute best Z across all species).  Elites are NOT mutated.
        all_survivors = [g for slist in survivors_per_species.values() for g in slist]
        all_survivors.sort(key=lambda g: g.fitness, reverse=True)
        elite_ids: set = set()
        for g in all_survivors[:n_elite]:
            child = g.clone()
            child.parent_species = g.parent_species
            child.id = self._next_gid
            self._next_gid += 1
            elite_ids.add(child.id)
            new_genomes.append(child)

        # --- Asexual reproduction per species ---
        # Distribute n_asexual across species proportional to species size
        total_members = sum(len(s) for s in survivors_per_species.values())
        if total_members == 0:
            # No survivors? Just keep current genomes
            return self.genomes[:pop_size]
        species_ids = list(survivors_per_species.keys())
        for sid in species_ids:
            members = survivors_per_species[sid]
            n_for_species = int(round(n_asexual * len(members) / total_members))
            for _ in range(n_for_species):
                if len(new_genomes) >= pop_size:
                    break
                parent = self._roulette(members, 1)[0]
                child = parent.clone()
                child.parent_species = sid
                child.id = self._next_gid
                self._next_gid += 1
                new_genomes.append(child)

        # --- Crossover within species ---
        for sid in species_ids:
            members = survivors_per_species[sid]
            n_for_species = int(round(n_crossover * len(members) / total_members))
            for _ in range(n_for_species):
                if len(new_genomes) >= pop_size:
                    break
                if len(members) < 2:
                    # Fall back to asexual
                    parent = self._roulette(members, 1)[0]
                    child = parent.clone()
                    child.parent_species = sid
                    child.id = self._next_gid
                    self._next_gid += 1
                    new_genomes.append(child)
                    continue
                parents = self._roulette(members, 2)
                if len(parents) < 2:
                    continue
                child = crossover(parents[0], parents[1], cfg, self.rng)
                child.parent_species = sid
                child.id = self._next_gid
                self._next_gid += 1
                new_genomes.append(child)

        # --- Interspecies crossover ---
        for _ in range(n_interspecies):
            if len(new_genomes) >= pop_size:
                break
            if len(species_ids) < 2:
                break
            sid1, sid2 = self.rng.choice(species_ids, size=2, replace=False)
            members1 = survivors_per_species.get(sid1, [])
            members2 = survivors_per_species.get(sid2, [])
            if not members1 or not members2:
                continue
            p1 = self._roulette(members1, 1)[0]
            p2 = self._roulette(members2, 1)[0]
            child = crossover(p1, p2, cfg, self.rng)
            child.parent_species = sid1
            child.id = self._next_gid
            self._next_gid += 1
            new_genomes.append(child)

        # --- Fill any remaining slots with random asexual reproduction ---
        while len(new_genomes) < pop_size:
            sid = self.rng.choice(species_ids)
            members = survivors_per_species.get(sid, [])
            if not members:
                # Pick any survivors
                members = all_survivors
            if not members:
                break
            parent = self._roulette(members, 1)[0]
            child = parent.clone()
            child.parent_species = sid
            child.id = self._next_gid
            self._next_gid += 1
            new_genomes.append(child)

        # Truncate if over
        return new_genomes[:pop_size], elite_ids

    # ------------------------------------------------------------------
    def _compute_stats(self) -> Dict[str, Any]:
        fits = np.array([g.fitness for g in self.genomes])
        n_conns = np.array([len(g.conns) for g in self.genomes])
        n_nodes = np.array([len(g.nodes) for g in self.genomes])
        return {
            "generation": self.generation,
            "pop_size": len(self.genomes),
            "n_species": len(self.speciator.species),
            "fitness_mean": float(fits.mean()) if len(fits) else 0.0,
            "fitness_max": float(fits.max()) if len(fits) else 0.0,
            "fitness_min": float(fits.min()) if len(fits) else 0.0,
            "fitness_std": float(fits.std()) if len(fits) else 0.0,
            "avg_conns": float(n_conns.mean()) if len(n_conns) else 0.0,
            "avg_nodes": float(n_nodes.mean()) if len(n_nodes) else 0.0,
            "max_conns": int(n_conns.max()) if len(n_conns) else 0,
            "max_nodes": int(n_nodes.max()) if len(n_nodes) else 0,
        }

    # ------------------------------------------------------------------
    def best(self) -> Optional[Genome]:
        if not self.genomes:
            return None
        return max(self.genomes, key=lambda g: g.fitness)

    def snapshot(self) -> Dict[str, Any]:
        """Lightweight snapshot for visualization / checkpointing."""
        return {
            "generation": self.generation,
            "stats": self.history[-1] if self.history else {},
            "species": [
                {"id": s.id, "members": len(s.members), "best_fitness": s.best_fitness}
                for s in self.speciator.species.values()
            ],
            "index": self.index.snapshot(),
        }
