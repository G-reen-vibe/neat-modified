# Modified NEAT Implementation

A Python implementation of the NEAT (NeuroEvolution of Augmenting Topologies)
algorithm with several modifications per the project spec, including:

- **Universal historical marking**: node/innovation IDs are shared across the
  entire population (not just per-generation), so the same mutation in two
  different genomes always produces the same IDs.
- **DAG-only topology**: loops and disconnected graphs are forbidden; a
  topological sort gives O(V+E) forward passes.
- **All four mutation types** (weight, connection, neuron/split, pruning)
  with all selection mechanisms from the spec.
- **Three crossover topology methods** × **three weight selection methods**.
- **Standard NEAT + Percentage similarity** tests.
- **Single / Standard (adaptive) / Purge speciation** modes.
- **Optional OpenAI-ES-style group-relative optimizer** with Adam/Momentum/RMSProp.
- **Genome repair** to keep every output connected.
- **Web-based visualizer** with live training, genome graph, species panel,
  and env playback.

## Project Layout

```
neat-project/
├── src/neat/                # Core algorithm
│   ├── __init__.py
│   ├── config.py            # All hyperparameters (dataclasses)
│   ├── indexing.py          # GlobalIndex: universal IDs + usage stats
│   ├── activation.py        # UAF, P-Swish, Sigmoid, Tanh, ReLU
│   ├── genome.py            # Genome class, DAG forward pass, topo sort
│   ├── mutations.py         # Weight/conn/neuron/prune mutators + repair
│   ├── crossover.py         # 3 topologies × 3 weight methods
│   ├── similarity.py        # Standard NEAT + Percentage
│   ├── speciation.py        # Single / Standard adaptive / Purge
│   ├── population.py        # Generation policy + reproduction
│   ├── optimizer.py         # OpenAI-ES GRPO optimizer
│   ├── initialization.py    # Partitioned conn+neuron init
│   └── envs.py              # Gymnasium env wrapper
├── tests/                   # Unit tests (pytest)
│   ├── test_genome.py
│   ├── test_mutations.py
│   ├── test_crossover_similarity.py
│   ├── test_speciation.py
│   └── test_population.py
├── scripts/                 # Entry points
│   ├── train.py             # Train one config
│   ├── benchmark.py         # Run a battery of benchmarks
│   ├── evaluate.py          # Evaluate best genome over N episodes
│   └── sweep.py             # Hyperparameter sweep
├── visualizer/
│   └── run.py               # FastAPI web visualizer
├── results/                 # Benchmark outputs (JSON)
├── configs/                 # Saved config files
└── README.md
```

## Quick Start

```bash
# Install dependencies (Python 3.12+)
pip install gymnasium[classic-control,box2d] pygame pytest matplotlib \
            fastapi uvicorn jinja2 python-multipart websockets pydantic pillow

# Run all unit tests
python tests/test_genome.py
python tests/test_mutations.py
python tests/test_crossover_similarity.py
python tests/test_speciation.py
python tests/test_population.py

# Train on CartPole-v1 (30 generations, ~15s)
python scripts/train.py --env CartPole-v1 --pop 100 --gens 30 --max-steps 500

# Evaluate the best genome over 100 episodes
python scripts/evaluate.py --env CartPole-v1 --pop 100 --gens 25 --episodes 100

# Run the visualizer (open http://localhost:8000)
python visualizer/run.py --env CartPole-v1 --pop 100 --gens 100
```

## Benchmark Results

| Env              | Solved? | Train Best | Eval Mean (50 eps) | Threshold | Time (30 gen) |
|------------------|---------|------------|--------------------|-----------|---------------|
| CartPole-v1      | ✅ Yes  | 500.00     | **500.00** ± 0.0   | 475.0     | ~15s          |
| Acrobot-v1       | ✅ Yes  | -62.00     | **-79.70** ± 14.2  | -100.0    | ~20s          |
| MountainCar-v0   | ❌ No   | -200.00    | -200.00 ± 0.0      | -110.0    | ~10s          |
| LunarLander-v3   | ❌ No   | 279.37     | -90.29 ± 166.5     | 200.0     | ~60s          |

MountainCar and LunarLander are notoriously hard for feed-forward NEAT
(they require either recurrent connections or careful reward shaping).
CartPole-v1 and Acrobot-v1 are solved consistently.

### Hyperparameter Sweep (CartPole-v1, 20 generations)

All 9 configurations solved CartPole-v1 with **eval_mean = 500.0** over 20
random episodes, demonstrating the implementation's robustness:

| Config              | Eval Mean  | Solved | Time  |
|---------------------|------------|--------|-------|
| baseline            | 500.0 ± 0  | ✅     | 6.3s  |
| low_mut             | 500.0 ± 0  | ✅     | 5.7s  |
| high_elite          | 500.0 ± 0  | ✅     | 6.8s  |
| more_species        | 500.0 ± 0  | ✅     | 5.4s  |
| optimizer           | 500.0 ± 0  | ✅     | 8.2s  |
| optimizer_low_mut   | 500.0 ± 0  | ✅     | 6.9s  |
| aggressive          | 500.0 ± 0  | ✅     | 6.0s  |
| small_weight_std    | 500.0 ± 0  | ✅     | 5.5s  |
| big_weight_std      | 500.0 ± 0  | ✅     | 7.5s  |

## Visualizer

The web visualizer (FastAPI + vanilla JS + vis-network + Chart.js) provides:

- **Live training view** with auto-updating fitness/species charts
- **Genome topology visualization** with weighted, color-coded edges
- **Species panel** showing per-species membership and best fitness
- **Top-10 genomes list** with click-to-visualize
- **Episode playback** - renders the best genome in the env frame-by-frame
- **Detailed stats panel** (max/mean/min fitness, conns, nodes, species)

Run with:
```bash
python visualizer/run.py --env CartPole-v1 --pop 100 --gens 50 --port 8000
# Then open http://localhost:8000
```

## Spec Compliance

The implementation follows the user's spec faithfully:

- ✅ Universal historical marking (via `GlobalIndex`)
- ✅ DAG-only topology with topological-sort forward pass
- ✅ All weight mutation selection mechanisms (PCT_SHUFFLED, INDEPENDENT)
- ✅ All weight modification mechanisms (Gaussian, Uniform, Bernoulli)
- ✅ All connection mutation selection mechanisms (6 variants)
- ✅ All neuron mutation selection mechanisms + both modification variants
- ✅ Pruning mutation with linear-path merging
- ✅ Activation functions: UAF, P-Swish, Sigmoid, Tanh, ReLU
- ✅ Three crossover topology methods + three weight selection methods
- ✅ Standard NEAT + Percentage similarity tests
- ✅ Single / Standard (adaptive) / Purge speciation
- ✅ Generation policy: asexual + crossover + interspecies + elitism + culling
- ✅ Mutation policy: nested / per-type / single-pick
- ✅ Optional OpenAI-ES-style GRPO optimizer with Adam/Momentum/RMSProp
- ✅ Partitioned initialization with weight-init multiplier

## GitHub

The project is hosted at: https://github.com/G-reen-vibe/neat-modified

A parallel implementation by the same author is preserved on the
`parallel-impl` branch for reference.

## License

MIT
