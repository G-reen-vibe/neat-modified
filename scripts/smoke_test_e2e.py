"""End-to-end smoke test: build a Population and run a few generations on a simple
fitness function (sum of |weights|, just to check the pipeline)."""
import sys
sys.path.insert(0, "/home/z/my-project/neat_python")

import numpy as np
from neat import Population, MutationPolicy, GRPOOptimizer
from neat.network import forward


def simple_fitness(genome):
    """Reward = -(sum of |weights|) so the optimizer prefers small weights,
    but also reward having at least 1 hidden node."""
    out = forward(genome, np.array([0.5, -0.5, 0.1, 0.2]))
    # reward: produce output near 0.5 from CartPole-like inputs
    return -abs(out[0] - 0.5) - 0.01 * sum(abs(c.weight) for c in genome.conns.values())


def main():
    pop = Population(
        n_inputs=4, n_outputs=1, size=20,
        optimizer=GRPOOptimizer(enabled=True, lr=0.05, weight_std=0.05,
                                 method="adam", similarity_method="percentage"),
        mutation_policy=MutationPolicy(
            weight_prob=1.0, connection_prob=0.3, neuron_prob=0.1, pruning_prob=0.05,
        ),
        speciation_policy="purge_then_standard",
        target_species=5,
        seed=42,
    )
    print(f"Initial population: {len(pop.genomes)} genomes")
    print(f"  best fitness: {pop.best_fitness}")
    for gen in range(5):
        stats = pop.step(simple_fitness)
        print(f"Gen {stats['generation']}: best={stats['best_fitness']:.3f} "
              f"mean={stats['mean_fitness']:.3f} species={stats['n_species']} "
              f"avg_conns={stats['avg_conns']:.1f} avg_nodes={stats['avg_nodes']:.1f}")
    print("\nE2E pipeline OK")


if __name__ == "__main__":
    main()
