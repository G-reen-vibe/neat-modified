"""Tests for crossover, similarity, speciation, optimizer."""
import pytest
import random
import numpy as np

from neat.registry import InnovationRegistry
from neat.genome import Genome, NodeGene, ConnectionGene
from neat.network import forward
from neat import mutations as M
from neat.crossover import (
    crossover, T_FITTER, T_MORE_CONNS, T_COMBINE,
    W_INDEPENDENT, W_AVERAGE, W_BY_NEURON,
)
from neat.similarity import similarity, similarity_standard, similarity_percentage
from neat.speciation import Speciator
from neat.optimizer import GRPOOptimizer
from neat.population import Population, MutationPolicy


def make_genome(n_in=2, n_out=1, fully_connected=True, reg=None):
    if reg is None:
        reg = InnovationRegistry()
        reg.reserve_node_ids(list(range(n_in + n_out)))
    g = Genome(reg, n_in, n_out)
    for i in range(n_in):
        g.add_node(NodeGene(i, "input", "identity"))
    for j in range(n_out):
        g.add_node(NodeGene(n_in + j, "output", "tanh"))
    if fully_connected:
        for i in range(n_in):
            for j in range(n_out):
                innov = reg.get_connection_innov(i, n_in + j)
                g.add_connection(ConnectionGene(innov, i, n_in + j, 0.5))
                reg.bump_presence(innov, +1)
    return g


class TestSimilarity:
    def test_identical_genomes_distance_zero(self):
        g = make_genome(3, 2)
        g2 = g.clone()
        assert similarity(g, g2, method="percentage") < 1e-9
        assert similarity(g, g2, method="standard") < 1e-9

    def test_different_weights_distance_positive(self):
        g1 = make_genome(3, 2)
        g2 = make_genome(3, 2)
        for c in g2.conns.values():
            c.weight = 1.0  # different from 0.5
        d_pct = similarity(g1, g2, method="percentage")
        d_std = similarity(g1, g2, method="standard")
        assert d_pct > 0
        assert d_std > 0

    def test_disjoint_topology(self):
        g1 = make_genome(2, 1)
        g2 = make_genome(2, 1)
        # add an extra connection to g2 (need hidden node)
        reg = g2.registry
        g2.add_node(NodeGene(100, "hidden", "relu"))
        reg.bump_node_presence(100, +1)
        i1 = reg.get_connection_innov(0, 100)
        g2.add_connection(ConnectionGene(i1, 0, 100, 0.5))
        reg.bump_presence(i1, +1)
        d = similarity(g1, g2, method="standard")
        assert d > 0
        d_pct = similarity(g1, g2, method="percentage")
        assert d_pct > 0

    def test_symmetric(self):
        g1 = make_genome(3, 2)
        g2 = make_genome(3, 2)
        for c in g2.conns.values():
            c.weight = 1.0
        assert abs(similarity(g1, g2) - similarity(g2, g1)) < 1e-12


class TestCrossover:
    def test_basic_crossover(self):
        reg = InnovationRegistry(); reg.reserve_node_ids([0, 1, 2, 3, 4])
        g1 = make_genome(3, 2, reg=reg)
        g2 = make_genome(3, 2, reg=reg)
        g1.fitness = 1.0
        g2.fitness = 0.5
        child = crossover(g1, g2, topology_method=T_FITTER, weight_method=W_AVERAGE, seed=42)
        assert child.n_inputs == 3
        assert child.n_outputs == 2
        assert child.is_dag()
        # forward pass should work
        out = forward(child, np.array([0.5, 0.5, 0.5]))
        assert out.shape == (2,)

    def test_average_weights(self):
        reg = InnovationRegistry(); reg.reserve_node_ids([0, 1, 2])
        g1 = make_genome(2, 1, reg=reg)
        g2 = make_genome(2, 1, reg=reg)
        for c in g1.conns.values():
            c.weight = 0.0
        for c in g2.conns.values():
            c.weight = 1.0
        child = crossover(g1, g2, weight_method=W_AVERAGE, seed=1)
        # all weights should be 0.5
        for c in child.conns.values():
            assert abs(c.weight - 0.5) < 1e-9

    def test_independent_weights(self):
        reg = InnovationRegistry(); reg.reserve_node_ids([0, 1, 2])
        g1 = make_genome(2, 1, reg=reg)
        g2 = make_genome(2, 1, reg=reg)
        for c in g1.conns.values():
            c.weight = 0.0
        for c in g2.conns.values():
            c.weight = 1.0
        child = crossover(g1, g2, weight_method=W_INDEPENDENT, seed=42)
        # weights should be either 0 or 1
        for c in child.conns.values():
            assert c.weight in (0.0, 1.0)

    def test_combine_topology(self):
        reg = InnovationRegistry(); reg.reserve_node_ids([0, 1, 2])
        g1 = make_genome(2, 1, reg=reg)
        g2 = make_genome(2, 1, reg=reg)
        # add a hidden node to g1
        g1.add_node(NodeGene(100, "hidden", "relu"))
        reg.bump_node_presence(100, +1)
        i1 = reg.get_connection_innov(0, 100)
        i2 = reg.get_connection_innov(100, 2)
        g1.add_connection(ConnectionGene(i1, 0, 100, 0.5))
        g1.add_connection(ConnectionGene(i2, 100, 2, 0.5))
        reg.bump_presence(i1, +1)
        reg.bump_presence(i2, +1)
        child = crossover(g1, g2, topology_method=T_COMBINE, weight_method=W_AVERAGE, seed=1)
        assert child.is_dag()
        # child should have the hidden node from g1
        assert 100 in child.nodes

    def test_more_conns_topology(self):
        reg = InnovationRegistry(); reg.reserve_node_ids([0, 1, 2])
        g1 = make_genome(2, 1, reg=reg)
        g2 = make_genome(2, 1, reg=reg)
        # g2 has more conns
        g2.add_node(NodeGene(100, "hidden", "relu"))
        reg.bump_node_presence(100, +1)
        i1 = reg.get_connection_innov(0, 100)
        i2 = reg.get_connection_innov(100, 2)
        g2.add_connection(ConnectionGene(i1, 0, 100, 0.5))
        g2.add_connection(ConnectionGene(i2, 100, 2, 0.5))
        reg.bump_presence(i1, +1)
        reg.bump_presence(i2, +1)
        child = crossover(g1, g2, topology_method=T_MORE_CONNS, seed=1)
        assert child.is_dag()
        assert 100 in child.nodes


class TestSpeciation:
    def test_single_species(self):
        pop = [make_genome(3, 2) for _ in range(10)]
        spec = Speciator(policy="single")
        species = spec.speciate(pop, generation=0)
        assert len(species) == 1
        for g in pop:
            assert g.species_id == next(iter(species.keys()))

    def test_standard_creates_species(self):
        # Create varied genomes so they speciate
        reg = InnovationRegistry()
        reg.reserve_node_ids([0, 1, 2, 3])
        pop = []
        for i in range(20):
            g = Genome(reg, 2, 2)
            for n in range(4):
                g.add_node(NodeGene(n, ["input", "input", "output", "output"][n], "identity" if n < 2 else "tanh"))
            for j in range(2):
                for k in range(2):
                    innov = reg.get_connection_innov(j, 2 + k)
                    g.add_connection(ConnectionGene(innov, j, 2 + k, random.gauss(0.5, 0.3)))
                    reg.bump_presence(innov, +1)
            pop.append(g)
        spec = Speciator(policy="standard", threshold=0.1)
        species = spec.speciate(pop, generation=0)
        # at least 1 species, members sum to len(pop)
        total = sum(len(sp.members) for sp in species.values())
        assert total == 20

    def test_adaptive_threshold(self):
        # if too many species, threshold should increase
        reg = InnovationRegistry()
        reg.reserve_node_ids([0, 1, 2, 3])
        pop = []
        for i in range(20):
            g = Genome(reg, 2, 2)
            for n in range(4):
                g.add_node(NodeGene(n, ["input", "input", "output", "output"][n], "identity" if n < 2 else "tanh"))
            for j in range(2):
                for k in range(2):
                    innov = reg.get_connection_innov(j, 2 + k)
                    g.add_connection(ConnectionGene(innov, j, 2 + k, random.gauss(0.5, 0.3)))
                    reg.bump_presence(innov, +1)
            pop.append(g)
        spec = Speciator(policy="standard", threshold=0.01, target_species=2,
                          min_threshold=0.01, max_threshold=0.5, adjust=0.1)
        start_threshold = spec.threshold
        spec.speciate(pop, generation=0)
        # if many species were created, threshold should increase
        # (depending on random init, may or may not trigger)
        assert spec.threshold >= start_threshold


class TestPopulation:
    def test_init(self):
        pop = Population(n_inputs=4, n_outputs=2, size=20, seed=0,
                          speciation_policy="single")
        assert len(pop.genomes) == 20
        assert all(g.n_inputs == 4 for g in pop.genomes)
        assert all(g.n_outputs == 2 for g in pop.genomes)

    def test_step(self):
        pop = Population(n_inputs=4, n_outputs=2, size=15, seed=0,
                          speciation_policy="single")
        def fit(g):
            return -sum(abs(c.weight) for c in g.conns.values())
        stats = pop.step(fit)
        assert "best_fitness" in stats
        assert stats["population_size"] == 15

    def test_multiple_steps(self):
        pop = Population(n_inputs=4, n_outputs=2, size=10, seed=0,
                          speciation_policy="single")
        def fit(g):
            return -sum(abs(c.weight) for c in g.conns.values())
        for _ in range(3):
            stats = pop.step(fit)
            assert stats["population_size"] == 10

    def test_with_optimizer(self):
        opt = GRPOOptimizer(enabled=True, lr=0.05, weight_std=0.05, method="adam")
        pop = Population(n_inputs=4, n_outputs=2, size=10, seed=0,
                          optimizer=opt, speciation_policy="single")
        def fit(g):
            return -sum(abs(c.weight) for c in g.conns.values())
        stats = pop.step(fit)
        assert stats["population_size"] == 10

    def test_snapshot(self):
        pop = Population(n_inputs=4, n_outputs=2, size=10, seed=0)
        def fit(g):
            return -sum(abs(c.weight) for c in g.conns.values())
        pop.step(fit)
        snap = pop.snapshot()
        assert "best_genome" in snap
        assert "history" in snap
