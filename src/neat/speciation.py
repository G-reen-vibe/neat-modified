"""
Speciation.

Per spec, three modes:
    Single   - all genomes in one species (testing)
    Standard - NEAT speciation with adaptive threshold:
        * If #species > target: increase threshold (merge more readily)
        * If #species < target: decrease threshold (split more readily)
        * First check the species a genome descended from (optimization)
        * When a new species is created, transfer info
    Purge    - first generation only:
        * Keep best X genomes, duplicate each into its own species with Y extra
          mutations, then compute an ideal threshold for those X species.

A species is represented as a dict with:
    id            - unique species id
    representative - genome used for similarity comparison (set once, left
                     unchanged so the species naturally diverges per NEAT paper)
    members       - list of genome ids
    best_fitness  - historical best fitness (for stagnation if needed)
    threshold     - current similarity threshold for membership (only the
                     global threshold is used here)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .config import Config, SpeciationKind, SimilarityKind
from .genome import Genome
from .indexing import GlobalIndex
from .similarity import similarity


class Species:
    def __init__(self, sid: int, representative: Genome):
        self.id = sid
        self.representative = representative
        self.members: List[int] = []   # genome ids
        self.best_fitness: float = representative.fitness
        self.last_improved: int = 0   # generation since last improvement

    def add(self, gid: int) -> None:
        self.members.append(gid)

    def clear(self) -> None:
        self.members = []

    def __repr__(self) -> str:
        return f"<Species {self.id} members={len(self.members)} best={self.best_fitness:.3f}>"


class Speciator:
    """Manages species across generations."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.species: Dict[int, Species] = {}
        self._next_sid = 0
        self.threshold = (cfg.speciation.min_threshold + cfg.speciation.max_threshold) / 2
        self.generation = 0

    def _new_species(self, rep: Genome) -> Species:
        sid = self._next_sid
        self._next_sid += 1
        s = Species(sid, rep)
        self.species[sid] = s
        return s

    def speciate(self, genomes: List[Genome]) -> Dict[int, Species]:
        """Assign each genome to a species.  Returns the species dict."""
        cfg = self.cfg
        # Choose mode for this generation
        if self.generation == 0:
            mode = cfg.speciation.initial_kind
        else:
            mode = cfg.speciation.subsequent_kind

        if mode == SpeciationKind.SINGLE:
            return self._speciate_single(genomes)
        if mode == SpeciationKind.PURGE:
            return self._speciate_purge(genomes)
        if mode == SpeciationKind.STANDARD:
            return self._speciate_standard(genomes)
        raise ValueError(f"Unknown speciation mode: {mode}")

    # ------------------------------------------------------------------
    def _speciate_single(self, genomes: List[Genome]) -> Dict[int, Species]:
        if not self.species:
            rep = genomes[0] if genomes else None
            if rep is None:
                return self.species
            self._new_species(rep)
        # Reset members
        for s in self.species.values():
            s.clear()
        only_sid = next(iter(self.species))
        for g in genomes:
            g.parent_species = only_sid
            self.species[only_sid].add(g.id)
        return self.species

    # ------------------------------------------------------------------
    def _speciate_standard(self, genomes: List[Genome]) -> Dict[int, Species]:
        """Standard NEAT speciation with adaptive threshold."""
        cfg = self.cfg

        # Clear members of all existing species (representatives stay)
        for s in self.species.values():
            s.clear()

        # First pass: try to assign each genome to its parent species
        # (optimization per the spec)
        unassigned: List[Genome] = []
        for g in genomes:
            placed = False
            if g.parent_species is not None and g.parent_species in self.species:
                s = self.species[g.parent_species]
                d = similarity(g, s.representative, cfg)
                if d <= self.threshold:
                    s.add(g.id)
                    placed = True
            if not placed:
                unassigned.append(g)

        # Second pass: try every other species
        still_unassigned: List[Genome] = []
        for g in unassigned:
            placed = False
            for s in self.species.values():
                d = similarity(g, s.representative, cfg)
                if d <= self.threshold:
                    g.parent_species = s.id
                    s.add(g.id)
                    placed = True
                    break
            if not placed:
                still_unassigned.append(g)

        # Third pass: create new species for the rest
        for g in still_unassigned:
            s = self._new_species(g)
            g.parent_species = s.id
            s.add(g.id)

        # ---- Adaptive threshold ----
        # Per spec: "If there are extra species above a set threshold, try to
        # merge any species that are too similar".  We make the adaptation
        # *direct*: if we have N species and want T, compute the (T)-th smallest
        # pairwise representative distance and use that as the new threshold,
        # so the next round of merging gives us approximately T species.
        n_species = len(self.species)
        target = cfg.speciation.target_species
        if n_species > target:
            # Compute pairwise distances between representatives
            reps = [s.representative for s in self.species.values()]
            if len(reps) >= 2:
                dists = []
                for i in range(len(reps)):
                    for j in range(i+1, len(reps)):
                        dists.append(similarity(reps[i], reps[j], cfg))
                dists.sort()
                # The threshold that would merge us down to ~target species:
                # we want to keep merging the closest pairs until we have target.
                # The (n_species - target)-th smallest distance is the threshold.
                idx = min(len(dists) - 1, max(0, n_species - target))
                new_thresh = dists[idx]
                # Move threshold toward this value (with some smoothing)
                self.threshold = min(cfg.speciation.max_threshold,
                                     max(cfg.speciation.min_threshold,
                                         0.5 * self.threshold + 0.5 * new_thresh))
            # Always try to merge similar species
            self._merge_similar()
        elif n_species < target:
            # Lower threshold to encourage splitting
            self.threshold = max(cfg.speciation.min_threshold,
                                 self.threshold - cfg.speciation.threshold_step)

        # Update best fitness
        for s in self.species.values():
            if s.members:
                fit_max = max(self._lookup(gid, genomes).fitness for gid in s.members) \
                          if genomes else 0.0
                if fit_max > s.best_fitness:
                    s.best_fitness = fit_max
                    s.last_improved = self.generation
        return self.species

    def _merge_similar(self) -> None:
        """Merge species whose representatives are within the threshold.

        Also, if we still have more than 2x the target species, keep merging
        the closest pairs until we get down to target.
        """
        cfg = self.cfg
        target = cfg.speciation.target_species

        # First, do threshold-based merging
        sids = list(self.species.keys())
        merged: Set[int] = set()
        for i, sid1 in enumerate(sids):
            if sid1 in merged:
                continue
            s1 = self.species[sid1]
            for sid2 in sids[i+1:]:
                if sid2 in merged:
                    continue
                s2 = self.species[sid2]
                d = similarity(s1.representative, s2.representative, cfg)
                if d < self.threshold:
                    for gid in s2.members:
                        s1.add(gid)
                    merged.add(sid2)
        for sid in merged:
            del self.species[sid]

        # If still way too many, iteratively merge closest pairs
        while len(self.species) > target:
            sids = list(self.species.keys())
            best_d = float("inf")
            best_pair = None
            for i in range(len(sids)):
                for j in range(i+1, len(sids)):
                    s1 = self.species[sids[i]]
                    s2 = self.species[sids[j]]
                    d = similarity(s1.representative, s2.representative, cfg)
                    if d < best_d:
                        best_d = d
                        best_pair = (sids[i], sids[j])
            if best_pair is None:
                break
            # Merge best_pair[1] into best_pair[0]
            s1 = self.species[best_pair[0]]
            s2 = self.species[best_pair[1]]
            for gid in s2.members:
                s1.add(gid)
            del self.species[best_pair[1]]
            # Update threshold to reflect this merge distance
            self.threshold = max(self.threshold, best_d)

    def _lookup(self, gid: int, genomes: List[Genome]) -> Genome:
        for g in genomes:
            if g.id == gid:
                return g
        raise KeyError(f"Genome {gid} not found")

    # ------------------------------------------------------------------
    def _speciate_purge(self, genomes: List[Genome]) -> Dict[int, Species]:
        """Purge mode: keep best X, duplicate into X species with Y extra mutations.

        Returns the species dict.  The caller (Population) is responsible for
        actually expanding the population back to pop_size in the generation
        policy; here we just set up the X species.

        After speciation, we compute the threshold as the *minimum* distance
        between any two of the X representatives (so they all stay in their
        own species under standard speciation).
        """
        cfg = self.cfg
        keep = cfg.speciation.purge_keep
        extra_mut = cfg.speciation.purge_extra_mutations
        # Sort by fitness desc and keep the top `keep`
        sorted_g = sorted(genomes, key=lambda g: g.fitness, reverse=True)
        kept = sorted_g[:keep]
        # Clear existing species
        self.species.clear()
        # Each kept genome gets its own species; we duplicate the genome with
        # `extra_mut` mutations applied as the species representative.
        rng = np.random.default_rng(self.cfg.seed + 9999)
        for g in kept:
            rep = g.clone()
            from .mutations import apply_mutation_policy
            for _ in range(extra_mut):
                apply_mutation_policy(rep, cfg, rng)
            s = self._new_species(rep)
            g.parent_species = s.id
            s.add(g.id)
        # Compute threshold: smallest pairwise distance among representatives
        reps = [s.representative for s in self.species.values()]
        if len(reps) >= 2:
            min_d = float("inf")
            for i in range(len(reps)):
                for j in range(i+1, len(reps)):
                    d = similarity(reps[i], reps[j], cfg)
                    if d < min_d:
                        min_d = d
            self.threshold = max(cfg.speciation.min_threshold,
                                 min(cfg.speciation.max_threshold, min_d * 1.5))
        return self.species

    # ------------------------------------------------------------------
    def advance_generation(self) -> None:
        self.generation += 1
