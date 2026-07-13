"""
Activation functions used by nodes.

Supported:
  * UAF (Universal Activation Function - a learnable parametric family
    described as a *linear combination* of common basis functions; for
    simplicity we approximate UAF as a "swish-like" parametric gate that
    the genome stores per-node, but here we expose only the standard
    fixed activations; UAF is implemented as the P-Swish with P learnable
    via the optimizer if requested).
  * P-Swish:  x * sigmoid(P * x).  P=0 reduces to identity (x * 0.5).
  * Sigmoid
  * Tanh
  * ReLU
  * identity (for inputs / bias)
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def identity(x: np.ndarray) -> np.ndarray:
    return x


def p_swish(x: np.ndarray, p: float) -> np.ndarray:
    """x * sigmoid(p * x).  p=0 -> x * 0.5 (a scaled identity)."""
    if p == 0.0:
        return 0.5 * x
    return x * sigmoid(p * x)


def uaf(x: np.ndarray, p: float = 1.0) -> np.ndarray:
    """Universal Activation Function.

    Implemented as P-Swish with parameter ``p`` (the genome controls ``p``).
    Keeping the same signature lets us swap UAF and P-Swish freely.
    """
    return p_swish(x, p)


# Mapping from name -> (callable, has_param)
ACTIVATIONS: dict = {
    "identity": (identity, False),
    "sigmoid": (sigmoid, False),
    "tanh": (tanh, False),
    "relu": (relu, False),
    "p_swish": (p_swish, True),
    "uaf": (uaf, True),
}


def get_activation(name: str) -> tuple:
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}")
    return ACTIVATIONS[name]
