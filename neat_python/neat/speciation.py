"""
Speciation policies.

Single    - all genomes in one species.
Standard  - NEAT speciation with adaptive threshold.
Purge     - first-generation: keep top X, duplicate into X species with mutations,
            then compute a similarity threshold that keeps them separated.

A Species holds:
  * id
  * representative genome (chosen once and then left unchanged per NEAT paper)
  * member genome list
  * a "staleness" counter (generations since fitness improved)
  * best fitness ever achieved
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .genome import Genome
from .similarity import similarity


@dataclass
class Species:
    species_id: int
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    best_fitness: float = -float("inf")
    staleness: int = 0
    # cached stats
    last_avg_fitness: float = 0.0

    def reset_members(self) -> None:
        self.members = []

    def add(self, g: Genome) -> None:
        g.species_id = self.species_id
        self.members.append(g)

    def update_stats(self) -> float:
        if not self.members:
            self.last_avg_fitness = 0.0
            return 0.0
        fits = [m.fitness for m in self.members]
        avg = sum(fits) / len(fits)
        self.last_avg_fitness = avg
        best = max(fits)
        if best > self.best_fitness:
            self.best_fitness = best
            self.staleness = 0
        else:
            self.staleness += 1
        return avg


class Speciator:
    """Implements Single / Standard / Purge speciation."""

    def __init__(
        self,
        policy: str = "standard",  # "single" | "standard" | "purge_then_standard"
        target_species: int = 10,
        threshold: float = 0.3,
        min_threshold: float = 0.05,
        max_threshold: float = 0.5,
        adjust: float = 0.025,
        similarity_method: str = "percentage",
    ) -> None:
        self.policy = policy
        self.target_species = target_species
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adjust = adjust
        self.similarity_method = similarity_method
        self._next_species_id = 0
        self.species: Dict[int, Species] = {}
        self._purge_done = False

    def _new_species_id(self) -> int:
        sid = self._next_species_id
        self._next_species_id += 1
        return sid

    def _dist(self, a: Genome, b: Genome) -> float:
        return similarity(a, b, method=self.similarity_method)

    # ------------------------------------------------------ single ---------
    def speciate_single(self, population: List[Genome]) -> Dict[int, Species]:
        if not self.species:
            sid = self._new_species_id()
            sp = Species(sid, population[0])
            self.species[sid] = sp
        for sp in self.species.values():
            sp.reset_members()
        only_species = next(iter(self.species.values()))
        only_species.representative = population[0]
        for g in population:
            only_species.add(g)
        return self.species

    # ------------------------------------------------------ purge ----------
    def speciate_purge(self, population: List[Genome], top_k: int, n_extra_mutations: int = 3, rng: Optional[random.Random] = None) -> Dict[int, Species]:
        """Purge all but the top_k genomes; duplicate each into its own species
        with extra mutations. Compute a threshold that keeps them apart."""
        rng = rng or random.Random()
        # sort by fitness desc
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        top = sorted_pop[:top_k]
        # build species around each top genome
        self.species.clear()
        # also clear registry presence counts and re-track
        for g in population:
            for c in g.conns.values():
                g.registry.bump_presence(c.innov, -1)
            for nid, n in g.nodes.items():
                if n.kind == "hidden":
                    g.registry.bump_node_presence(nid, -1)
        new_pop: List[Genome] = []
        for g in top:
            sid = self._new_species_id()
            sp = Species(sid, g)
            self.species[sid] = sp
            sp.add(g)
            for c in g.conns.values():
                g.registry.bump_presence(c.innov, +1)
            for nid, n in g.nodes.items():
                if n.kind == "hidden":
                    g.registry.bump_node_presence(nid, +1)
            new_pop.append(g)
            # create n_extra_mutations children of g in the same species
            from . import mutations as M
            for k in range(n_extra_mutations):
                child = g.clone()
                child._undo_log = []
                # apply a few random mutations
                M.mutate_weights(child, {"selection": M.W_SELECT_PROB, "pct": 0.5,
                                          "mod": M.M_GAUSSIAN, "mod_param": 0.1}, seed=rng.randrange(1 << 30))
                M.mutate_add_connection(child, {"selection": M.C_SELECT_PCT_SHUFFLED,
                                                 "pct": 0.0, "mod": M.M_GAUSSIAN,
                                                 "mod_param": 0.2}, seed=rng.randrange(1 << 30))
                sp.add(child)
                for c in child.conns.values():
                    child.registry.bump_presence(c.innov, +1)
                for nid, n in child.nodes.items():
                    if n.kind == "hidden":
                        child.registry.bump_node_presence(nid, +1)
                new_pop.append(child)
        # replace the population list (caller should use new_pop)
        population[:] = new_pop
        # compute threshold: max pairwise distance within species, min pairwise across
        # we want threshold that keeps the top genomes in their own species
        # i.e. larger than any in-species distance, smaller than any cross-species distance
        # for simplicity, take the average cross-species distance / 2
        reps = [sp.representative for sp in self.species.values()]
        if len(reps) >= 2:
            dists = []
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    dists.append(self._dist(reps[i], reps[j]))
            if dists:
                # set threshold so that the closest cross-species pair is at the boundary
                self.threshold = max(self.min_threshold, min(self.max_threshold, min(dists) * 0.5))
        self._purge_done = True
        return self.species

    # ------------------------------------------------------ standard -------
    def speciate_standard(self, population: List[Genome]) -> Dict[int, Species]:
        # reset members
        for sp in self.species.values():
            sp.reset_members()
        # First, try to match each genome to an existing species (by representative).
        # Optimize: check the species a genome descended from first.
        for g in population:
            placed = False
            # try the genome's previous species first
            if g.species_id in self.species:
                sp = self.species[g.species_id]
                d = self._dist(g, sp.representative)
                if d < self.threshold:
                    sp.add(g)
                    placed = True
            if not placed:
                # try all species in order
                for sp in self.species.values():
                    d = self._dist(g, sp.representative)
                    if d < self.threshold:
                        sp.add(g)
                        placed = True
                        break
            if not placed:
                # create a new species with this genome as representative
                sid = self._new_species_id()
                sp = Species(sid, g)
                self.species[sid] = sp
                sp.add(g)

        # remove empty species
        empty = [sid for sid, sp in self.species.items() if not sp.members]
        for sid in empty:
            del self.species[sid]

        # adaptive threshold: if too many species, increase threshold & try to merge
        if len(self.species) > self.target_species:
            self.threshold = min(self.max_threshold, self.threshold + self.adjust)
            self._try_merge_species()
        elif len(self.species) < self.target_species * 0.5:
            self.threshold = max(self.min_threshold, self.threshold - self.adjust)

        return self.species

    def _try_merge_species(self) -> None:
        """Merge species whose representatives are within threshold."""
        sids = list(self.species.keys())
        merged = set()
        for i in range(len(sids)):
            if sids[i] in merged:
                continue
            for j in range(i + 1, len(sids)):
                if sids[j] in merged:
                    continue
                a = self.species[sids[i]]
                b = self.species[sids[j]]
                d = self._dist(a.representative, b.representative)
                if d < self.threshold:
                    # merge b into a
                    for g in b.members:
                        a.add(g)
                    merged.add(sids[j])
        for sid in merged:
            del self.species[sid]

    # ------------------------------------------------------ dispatch -------
    def speciate(self, population: List[Genome], generation: int, top_k_purge: int = 5) -> Dict[int, Species]:
        if self.policy == "single":
            return self.speciate_single(population)
        if self.policy == "purge_then_standard":
            if generation == 0 and not self._purge_done:
                return self.speciate_purge(population, top_k=top_k_purge)
            return self.speciate_standard(population)
        if self.policy == "standard":
            return self.speciate_standard(population)
        raise ValueError(f"unknown speciation policy {self.policy}")
