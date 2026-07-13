"""
OpenAI-ES-style group-relative optimizer (per spec, "Optimizer (Optional)").

Algorithm (paraphrasing the spec):
1.  For each genome G in species S, compute its relative improvement:
        ri_G = (reward_G - mean(reward_S)) / std(reward_S)
2.  The *partial gradient* contributed by G is:
        pg_G = delta_G * ri_G
    where delta_G is the *weight mutation delta* from G's last weight mutation.
3.  Partial gradients are computed BEFORE culling, and applied AFTER culling
    and BEFORE reproduction.  If we're using partial gradients, we also
    *revert* the weight mutation that produced delta_G.
4.  The *applied gradient* to a single genome G' is:
        ag_G' = sum over G in S of  similarity(G, G') * pg_G
    So more similar genomes share more gradient.
5.  Normalize by the total mass of the partial gradients (similarity sum),
    multiply by learning_rate and by weight_std (the spec notes weight_std
    plays the role of the inverse Fisher information matrix).
6.  Apply weight decay (L1 and L2).
7.  Optionally use ADAM / Momentum / RMSProp, where the first/second moments
    are stored *per-connection* and inherited/averaged during crossover.

For activation parameters (UAF, P-Swish), we apply a similar update using
the activation deltas.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import Config, OptimizerCfg
from .genome import Genome, Connection
from .indexing import GlobalIndex
from .similarity import similarity_percentage


class Optimizer:
    """Implements the OpenAI-ES-style group-relative gradient step."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.opt = cfg.optimizer
        self.t = 0   # timestep for ADAM bias correction

    # ------------------------------------------------------------------
    def step(self, species_members: List[Genome],
             all_genomes: List[Genome]) -> None:
        """Apply one optimizer step to the genomes in `species_members`.

        Per spec: partial gradients are computed BEFORE culling.  We assume
        the caller has already evaluated fitness on these genomes.  We:
          1. Compute the species reward mean & std.
          2. For each genome, get its weight-mutation deltas and compute
             partial gradient = delta * relative_improvement.
          3. Revert the weight mutation (restore pre-mutation weights).
          4. For each genome, compute applied gradient as the similarity-
             weighted sum of partial gradients, normalize, multiply by lr
             and weight_std.
          5. Apply optimizer update (ADAM / Momentum / RMSProp / SGD) using
             the per-connection m, v stored on the connection.
        """
        if not self.opt.enabled or len(species_members) < 2:
            # Still revert weight mutations even if not optimizing
            for g in species_members:
                self._revert_weight_mutation(g)
            return

        rewards = np.array([g.fitness for g in species_members], dtype=np.float64)
        mu = rewards.mean()
        sigma = rewards.std()
        if sigma < 1e-8:
            sigma = 1.0
        ri = (rewards - mu) / sigma   # relative improvement per genome

        # 1. Compute partial gradients per genome (innov -> delta * ri)
        #    Also collect the set of all innovations touched.
        partial_grads: List[Dict[int, float]] = []
        all_innovs: set = set()
        for g, r in zip(species_members, ri):
            deltas = g.consume_weight_deltas()
            pg = {iv: d * r for iv, d in deltas.items()}
            partial_grads.append(pg)
            all_innovs.update(pg.keys())

        # 2. Compute pairwise similarity between genomes that contributed a
        #    partial gradient (only among species_members, since spec says
        #    "similarity to each other genome within the species").
        n = len(species_members)
        sim_matrix = np.ones((n, n), dtype=np.float64)  # diagonal = 1
        for i in range(n):
            for j in range(i+1, n):
                d = similarity_percentage(species_members[i], species_members[j], self.cfg)
                s = 1.0 - d
                sim_matrix[i, j] = s
                sim_matrix[j, i] = s

        # 3. Revert weight mutations (restore pre-mutation weights) for all
        #    genomes that contributed a partial gradient.
        for g, pg in zip(species_members, partial_grads):
            self._revert_weight_mutation(g, pg)

        # 4. For each genome, compute the applied gradient per innov and
        #    apply the optimizer step.
        self.t += 1
        lr = self.opt.lr
        std = self.cfg.mutation.weight_std
        l2 = self.opt.weight_decay_l2
        l1 = self.opt.weight_decay_l1

        # Total mass of partial gradients (for normalization)
        total_mass = sum(sum(abs(v) for v in pg.values()) for pg in partial_grads)
        if total_mass < 1e-12:
            total_mass = 1.0

        for i, g in enumerate(species_members):
            # Applied gradient per innov
            applied: Dict[int, float] = {}
            for j, pg_j in enumerate(partial_grads):
                s_ij = sim_matrix[i, j]
                if s_ij <= 0:
                    continue
                for iv, val in pg_j.items():
                    applied[iv] = applied.get(iv, 0.0) + s_ij * val
            # Normalize, scale, apply
            for iv, grad in applied.items():
                if iv not in g.conns:
                    continue
                c = g.conns[iv]
                grad = grad / total_mass * lr * std
                # Add weight decay (gradient of L2 is 2*w, of L1 is sign(w))
                grad = grad - l2 * c.weight - l1 * np.sign(c.weight)
                # Optimizer-specific update
                if self.opt.method == "sgd":
                    c.weight = c.weight + grad
                elif self.opt.method == "momentum":
                    c.m = self.opt.momentum * c.m + grad
                    c.weight = c.weight + c.m
                elif self.opt.method == "rmsprop":
                    c.v = self.opt.rmsprop_decay * c.v + (1 - self.opt.rmsprop_decay) * grad * grad
                    c.weight = c.weight + grad / (np.sqrt(c.v) + self.opt.adam_eps)
                elif self.opt.method == "adam":
                    c.m = self.opt.adam_beta1 * c.m + (1 - self.opt.adam_beta1) * grad
                    c.v = self.opt.adam_beta2 * c.v + (1 - self.opt.adam_beta2) * grad * grad
                    m_hat = c.m / (1 - self.opt.adam_beta1 ** self.t)
                    v_hat = c.v / (1 - self.opt.adam_beta2 ** self.t)
                    c.weight = c.weight + m_hat / (np.sqrt(v_hat) + self.opt.adam_eps)
                else:
                    raise ValueError(f"Unknown optimizer method: {self.opt.method}")

    # ------------------------------------------------------------------
    def _revert_weight_mutation(self, g: Genome,
                                 pg: Optional[Dict[int, float]] = None) -> None:
        """Restore pre-mutation weights using the recorded deltas.

        If `pg` (partial gradients) is provided, we use its keys (which
        contain the deltas); otherwise we use g._last_weight_mut_delta.
        """
        if pg is None:
            pg = g.consume_weight_deltas()
        else:
            # consume them so they don't get used later
            g.consume_weight_deltas()
        for iv, delta in pg.items():
            if iv in g.conns:
                g.conns[iv].weight -= delta
