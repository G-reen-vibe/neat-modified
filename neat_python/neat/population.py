"""
Population: ties together mutation policy, generation policy, speciation,
and the optimizer.

The lifecycle per generation:
  1. evaluate fitness (caller-provided function)
  2. speciate
  3. within each species:
     a. compute species stats
     b. GRPO optimizer step (reverts weight mutations)
     c. cull bottom X%
     d. elitism (carry top Z unchanged)
     e. fill remainder via asexual reproduction + mutation, and crossover
  4. interspecies crossovers (W of them)
  5. apply mutation policy to all children
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .genome import Genome, NodeGene, ConnectionGene
from . import mutations as M
from .crossover import crossover, T_FITTER, W_AVERAGE
from .speciation import Speciator, Species
from .optimizer import GRPOOptimizer
from .similarity import similarity


# Mutation policy selection methods
MP_NESTED = 0          # nest several mutation policies with step durations
MP_PER_TYPE_PROB = 1   # check prob for each mutation type, apply all picked
MP_SINGLE_PICK = 2     # single probability choice; pick one (or none)


class MutationPolicy:
    """Encapsulates the mutation policy.

    For MP_PER_TYPE_PROB (default): each mutation type is checked independently.
      pruning_prob, neuron_prob, connection_prob, weight_prob

    For MP_SINGLE_PICK: pick one mutation type to apply (or none).

    For MP_NESTED: cycle through sub-policies; each lasts ``step_duration``
    generations before switching. Useful for phased pruning.
    """

    def __init__(
        self,
        method: int = MP_PER_TYPE_PROB,
        pruning_prob: float = 0.05,
        neuron_prob: float = 0.10,
        connection_prob: float = 0.25,
        weight_prob: float = 1.0,
        # sub-configs (passed straight to mutation functions)
        weight_cfg: Optional[Dict] = None,
        connection_cfg: Optional[Dict] = None,
        neuron_cfg: Optional[Dict] = None,
        pruning_cfg: Optional[Dict] = None,
        # nested
        sub_policies: Optional[List[Tuple["MutationPolicy", int]]] = None,
    ) -> None:
        self.method = method
        self.pruning_prob = pruning_prob
        self.neuron_prob = neuron_prob
        self.connection_prob = connection_prob
        self.weight_prob = weight_prob
        self.weight_cfg = weight_cfg or {
            "selection": M.W_SELECT_PCT_SHUFFLED, "pct": 1.0,
            "mod": M.M_GAUSSIAN, "mod_param": 0.05,
        }
        self.connection_cfg = connection_cfg or {
            "selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
            "mod": M.M_GAUSSIAN, "mod_param": 0.2,
        }
        self.neuron_cfg = neuron_cfg or {
            "selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
            "mod": M.N_SPLIT_INCOMING_ONE,
        }
        self.pruning_cfg = pruning_cfg or {
            "selection": M.P_SELECT_PCT_SHUFFLED, "pct": 0.0,
        }
        self.sub_policies = sub_policies or []
        self._nested_idx = 0
        self._nested_step = 0

    def apply(self, genome: Genome, rng: random.Random, species_members: Optional[List[Genome]] = None) -> None:
        if self.method == MP_NESTED:
            return self._apply_nested(genome, rng, species_members)
        if self.method == MP_PER_TYPE_PROB:
            return self._apply_per_type(genome, rng, species_members)
        if self.method == MP_SINGLE_PICK:
            return self._apply_single(genome, rng, species_members)
        raise ValueError(f"unknown mutation policy method {self.method}")

    def _apply_nested(self, genome: Genome, rng: random.Random, species_members: Optional[List[Genome]] = None) -> None:
        if not self.sub_policies:
            return
        sub, duration = self.sub_policies[self._nested_idx]
        sub.apply(genome, rng, species_members)
        self._nested_step += 1
        if self._nested_step >= duration:
            self._nested_step = 0
            self._nested_idx = (self._nested_idx + 1) % len(self.sub_policies)

    def _apply_per_type(self, genome: Genome, rng: random.Random, species_members: Optional[List[Genome]] = None) -> None:
        seed = rng.randrange(1 << 30)
        if rng.random() < self.pruning_prob:
            M.mutate_prune(genome, self.pruning_cfg, seed=seed)
            seed = rng.randrange(1 << 30)
        if rng.random() < self.neuron_prob:
            M.mutate_add_neuron(genome, self.neuron_cfg, seed=seed, species_members=species_members)
            seed = rng.randrange(1 << 30)
        if rng.random() < self.connection_prob:
            M.mutate_add_connection(genome, self.connection_cfg, seed=seed, species_members=species_members)
            seed = rng.randrange(1 << 30)
        if rng.random() < self.weight_prob:
            M.mutate_weights(genome, self.weight_cfg, seed=seed)

    def _apply_single(self, genome: Genome, rng: random.Random, species_members: Optional[List[Genome]] = None) -> None:
        # weighted pick
        types = []
        probs = []
        if self.pruning_prob > 0:
            types.append("prune"); probs.append(self.pruning_prob)
        if self.neuron_prob > 0:
            types.append("neuron"); probs.append(self.neuron_prob)
        if self.connection_prob > 0:
            types.append("conn"); probs.append(self.connection_prob)
        if self.weight_prob > 0:
            types.append("weight"); probs.append(self.weight_prob)
        # add 'none' option
        total = sum(probs)
        none_prob = max(0.0, 1.0 - total)
        r = rng.random()
        if r < none_prob:
            return
        # pick one proportional to probs
        r = (r - none_prob) / total if total > 0 else 0.0
        acc = 0.0
        chosen = types[0]
        for t, p in zip(types, probs):
            acc += p / total
            if r <= acc:
                chosen = t
                break
        seed = rng.randrange(1 << 30)
        if chosen == "prune":
            M.mutate_prune(genome, self.pruning_cfg, seed=seed)
        elif chosen == "neuron":
            M.mutate_add_neuron(genome, self.neuron_cfg, seed=seed, species_members=species_members)
        elif chosen == "conn":
            M.mutate_add_connection(genome, self.connection_cfg, seed=seed, species_members=species_members)
        elif chosen == "weight":
            M.mutate_weights(genome, self.weight_cfg, seed=seed)


# ----------------------------------------------------------------- Population
class Population:
    """The main NEAT population."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        size: int = 100,
        # initialization
        init_conns_multiplier: float = 2.0,
        init_neuron_range: Tuple[int, int] = (0, 2),
        # generation policy
        asexual_pct: float = 0.75,
        crossover_pct: float = 0.25,
        n_interspecies: int = 1,
        n_elitism: int = 1,
        cull_pct: float = 0.5,
        # crossover
        topology_method: int = T_FITTER,
        weight_method: int = W_AVERAGE,
        # speciation
        speciation_policy: str = "purge_then_standard",
        target_species: int = 10,
        threshold: float = 0.3,
        min_threshold: float = 0.05,
        max_threshold: float = 0.5,
        threshold_adjust: float = 0.025,
        similarity_method: str = "percentage",
        # optimizer
        optimizer: Optional[GRPOOptimizer] = None,
        # mutation policy
        mutation_policy: Optional[MutationPolicy] = None,
        # output activation override (default "tanh"; use "identity" for continuous control)
        output_activation: str = "tanh",
        # add a bias node (always outputs 1.0) connected to all outputs
        use_bias: bool = False,
        # seed
        seed: int = 0,
    ) -> None:
        from .registry import InnovationRegistry
        self.registry = InnovationRegistry()
        # reserve input/output node ids (and bias id if used)
        all_ids = list(range(n_inputs + n_outputs))
        if use_bias:
            bias_id = n_inputs + n_outputs
            all_ids.append(bias_id)
        self.registry.reserve_node_ids(all_ids)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.size = size
        self.init_conns_multiplier = init_conns_multiplier
        self.init_neuron_range = init_neuron_range
        self.asexual_pct = asexual_pct
        self.crossover_pct = crossover_pct
        self.n_interspecies = n_interspecies
        self.n_elitism = n_elitism
        self.cull_pct = cull_pct
        self.topology_method = topology_method
        self.weight_method = weight_method
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.optimizer = optimizer or GRPOOptimizer(enabled=False, similarity_method=similarity_method)
        self.mutation_policy = mutation_policy or MutationPolicy()
        self.speciator = Speciator(
            policy=speciation_policy,
            target_species=target_species,
            threshold=threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            adjust=threshold_adjust,
            similarity_method=similarity_method,
        )
        self.generation: int = 0
        self.rng = random.Random(seed)
        self.genomes: List[Genome] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = -float("inf")
        self.history: List[Dict] = []
        self._init_population()

    # ----------------------------------------------------- initialization --
    def _make_initial_genome(self) -> Genome:
        """Build a starter genome: n_inputs -> n_outputs fully connected,
        with weight multiplier applied, plus 0-2 hidden nodes via neuron
        mutation (the spec says to partition init so connection mutations
        happen before each neuron mutation to avoid hanging neurons)."""
        g = Genome(self.registry, self.n_inputs, self.n_outputs)
        for i in range(self.n_inputs):
            g.add_node(NodeGene(i, "input", "identity"))
        # bias node (always outputs 1.0) — id = n_inputs + n_outputs
        bias_id = self.n_inputs + self.n_outputs
        if self.use_bias:
            g.add_node(NodeGene(bias_id, "bias", "identity"))
        for j in range(self.n_outputs):
            g.add_node(NodeGene(self.n_inputs + j, "output", self.output_activation))
        # number of weights: total possible = n_inputs * n_outputs (no extra nodes yet)
        # if bias: + n_outputs extra connections
        n_conns = int(self.n_inputs * self.n_outputs * self.init_conns_multiplier)
        base_conns = min(n_conns, self.n_inputs * self.n_outputs)
        for i in range(self.n_inputs):
            for j in range(self.n_outputs):
                if len(g.conns) >= base_conns:
                    break
                innov = self.registry.get_connection_innov(i, self.n_inputs + j)
                w = self.rng.gauss(0.0, 0.5) * self.init_conns_multiplier
                g.add_connection(ConnectionGene(innov, i, self.n_inputs + j, w))
                self.registry.bump_presence(innov, +1)
        # add bias->output connections
        if self.use_bias:
            for j in range(self.n_outputs):
                innov = self.registry.get_connection_innov(bias_id, self.n_inputs + j)
                w = self.rng.gauss(0.0, 0.5) * self.init_conns_multiplier
                g.add_connection(ConnectionGene(innov, bias_id, self.n_inputs + j, w))
                self.registry.bump_presence(innov, +1)
        # add 0-2 neurons via splitting
        n_neurons = self.rng.randint(*self.init_neuron_range)
        for _ in range(n_neurons):
            ok = M.mutate_add_neuron(
                g,
                {"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0, "mod": M.N_SPLIT_INCOMING_ONE},
                seed=self.rng.randrange(1 << 30),
            )
            if not ok:
                break
        return g

    def _init_population(self) -> None:
        self.genomes = [self._make_initial_genome() for _ in range(self.size)]
        for g in self.genomes:
            g.generation = 0

    # --------------------------------------------------------- evaluate ----
    def evaluate(self, fitness_fn: Callable[[Genome], float]) -> None:
        """Evaluate every genome's fitness (in-place)."""
        for g in self.genomes:
            g.fitness = fitness_fn(g)
            if g.fitness > self.best_fitness:
                self.best_fitness = g.fitness
                self.best_genome = g.clone()

    # --------------------------------------------------------- step --------
    def step(self, fitness_fn: Callable[[Genome], float]) -> Dict:
        """Run one full generation: evaluate, speciate, optimize, reproduce."""
        self.evaluate(fitness_fn)
        # speciate
        species_map = self.speciator.speciate(self.genomes, generation=self.generation)
        # update species stats
        for sp in species_map.values():
            sp.update_stats()
        # GRPO optimizer step (per species), which also reverts weight mutations
        for sp in species_map.values():
            self.optimizer.step(sp.members)
        # within each species: cull, elitism, reproduce
        new_population: List[Genome] = []
        # compute species offspring counts proportional to avg fitness
        total_avg = sum(sp.last_avg_fitness for sp in species_map.values())
        if total_avg <= 0:
            total_avg = 1.0
        # reserve space for interspecies & elitism
        reserved = self.n_interspecies + sum(min(self.n_elitism, len(sp.members)) for sp in species_map.values())
        remaining = max(0, self.size - reserved)
        for sp in species_map.values():
            # elitism: carry top Z unchanged
            elite_count = min(self.n_elitism, len(sp.members))
            sorted_members = sorted(sp.members, key=lambda g: g.fitness, reverse=True)
            elites = sorted_members[:elite_count]
            for e in elites:
                e_child = e.clone()
                e_child._undo_log = []
                new_population.append(e_child)
            # offspring count proportional to species avg fitness
            share = (sp.last_avg_fitness / total_avg) * remaining if total_avg > 0 else remaining / len(species_map)
            n_offspring = max(0, int(round(share)))
            # cull bottom cull_pct
            n_keep = max(1, int(len(sp.members) * (1.0 - self.cull_pct)))
            kept = sorted_members[:n_keep]
            if not kept:
                continue
            n_asexual = int(round(n_offspring * self.asexual_pct))
            n_cross = n_offspring - n_asexual
            # asexual
            for _ in range(n_asexual):
                parent = self._roulette_pick(kept)
                child = parent.clone()
                child._undo_log = []
                child.fitness = 0.0
                child.weight_delta = {}
                self.mutation_policy.apply(child, self.rng, species_members=kept)
                new_population.append(child)
            # crossover
            for _ in range(n_cross):
                p1 = self._roulette_pick(kept)
                p2 = self._roulette_pick(kept)
                while p2 is p1 and len(kept) > 1:
                    p2 = self._roulette_pick(kept)
                child = crossover(p1, p2,
                                  topology_method=self.topology_method,
                                  weight_method=self.weight_method,
                                  seed=self.rng.randrange(1 << 30))
                child.fitness = 0.0
                child.weight_delta = {}
                self.mutation_policy.apply(child, self.rng, species_members=kept)
                new_population.append(child)
        # interspecies crossovers
        species_list = list(species_map.values())
        for _ in range(self.n_interspecies):
            if len(species_list) < 2:
                break
            sp_a, sp_b = self.rng.sample(species_list, 2)
            if not sp_a.members or not sp_b.members:
                continue
            p1 = self.rng.choice(sp_a.members)
            p2 = self.rng.choice(sp_b.members)
            child = crossover(p1, p2,
                              topology_method=self.topology_method,
                              weight_method=self.weight_method,
                              seed=self.rng.randrange(1 << 30))
            child.fitness = 0.0
            child.weight_delta = {}
            self.mutation_policy.apply(child, self.rng, species_members=None)
            new_population.append(child)
        # cap at size
        if len(new_population) > self.size:
            new_population = new_population[:self.size]
        elif len(new_population) < self.size:
            # fill remainder with mutated clones of the best
            while len(new_population) < self.size:
                parent = self.best_genome or self.rng.choice(self.genomes)
                child = parent.clone()
                child._undo_log = []
                child.fitness = 0.0
                child.weight_delta = {}
                self.mutation_policy.apply(child, self.rng, species_members=None)
                new_population.append(child)
        # update presence counts: remove all old, add all new
        for g in self.genomes:
            for c in g.conns.values():
                self.registry.bump_presence(c.innov, -1)
            for nid, n in g.nodes.items():
                if n.kind == "hidden":
                    self.registry.bump_node_presence(nid, -1)
        for g in new_population:
            for c in g.conns.values():
                self.registry.bump_presence(c.innov, +1)
            for nid, n in g.nodes.items():
                if n.kind == "hidden":
                    self.registry.bump_node_presence(nid, +1)
        self.genomes = new_population
        self.generation += 1
        # history
        stats = {
            "generation": self.generation - 1,
            "best_fitness": self.best_fitness,
            "mean_fitness": sum(g.fitness for g in self.genomes) / max(1, len(self.genomes)),
            "n_species": len(species_map),
            "population_size": len(self.genomes),
            "avg_conns": sum(len(g.conns) for g in self.genomes) / max(1, len(self.genomes)),
            "avg_nodes": sum(len(g.nodes) for g in self.genomes) / max(1, len(self.genomes)),
            "species_threshold": self.speciator.threshold,
        }
        self.history.append(stats)
        return stats

    # --------------------------------------------------------- roulette ----
    def _roulette_pick(self, genomes: List[Genome]) -> Genome:
        fits = [max(0.0, g.fitness) for g in genomes]
        total = sum(fits)
        if total <= 0:
            return self.rng.choice(genomes)
        r = self.rng.random() * total
        acc = 0.0
        for g, f in zip(genomes, fits):
            acc += f
            if r <= acc:
                return g
        return genomes[-1]

    # --------------------------------------------------------- snapshot ----
    def snapshot(self) -> Dict:
        """Return a JSON-serializable snapshot of the population."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "history": self.history[-100:],
            "n_species": len(self.speciator.species),
            "threshold": self.speciator.threshold,
            "species": [
                {
                    "id": sp.species_id,
                    "size": len(sp.members),
                    "best_fitness": sp.best_fitness,
                    "staleness": sp.staleness,
                    "avg_fitness": sp.last_avg_fitness,
                    "representative": sp.representative.to_dict() if sp.representative else None,
                }
                for sp in self.speciator.species.values()
            ],
            "best_genome": self.best_genome.to_dict() if self.best_genome else None,
            "genomes": [g.to_dict() for g in sorted(self.genomes, key=lambda x: x.fitness, reverse=True)[:20]],
        }
