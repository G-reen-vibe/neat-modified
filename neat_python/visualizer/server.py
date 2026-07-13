"""
NEAT Visualizer Backend.

A FastAPI + WebSocket server that:
  * Runs the NEAT training loop in a background thread
  * Exposes HTTP endpoints for snapshots and control
  * Streams live updates via WebSocket
  * Serves episode replays for the live training view

Run:
    python3 server.py [--port 8000]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# add neat_python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym

from neat import (
    Population, MutationPolicy, GRPOOptimizer,
    MP_PER_TYPE_PROB, MP_SINGLE_PICK, MP_NESTED,
)
from neat import mutations as M
from neat.network import forward, act_discrete
from neat.genome import Genome


# ----------------------------------------------------------------- state ----
class TrainingState:
    def __init__(self) -> None:
        self.population: Optional[Population] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest_snapshot: Dict[str, Any] = {}
        self.episode_buffer: List[Dict] = []
        self.max_episode_buffer = 50
        self.config: Dict = {}
        self.stop_flag = threading.Event()
        self.subscribers: List[asyncio.Queue] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.generation_delay: float = 0.0


STATE = TrainingState()


# ----------------------------------------------------------------- helpers --
def build_population(config: Dict) -> Population:
    opt = GRPOOptimizer(
        enabled=config.get("optimizer_enabled", False),
        lr=config.get("opt_lr", 0.02),
        weight_std=config.get("weight_std", 0.05),
        l2=config.get("opt_l2", 0.0),
        method=config.get("opt_method", "adam"),
        similarity_method="percentage",
    )
    mut = MutationPolicy(
        method=MP_PER_TYPE_PROB,
        weight_prob=config.get("weight_prob", 0.8),
        connection_prob=config.get("connection_prob", 0.1),
        neuron_prob=config.get("neuron_prob", 0.05),
        pruning_prob=config.get("pruning_prob", 0.02),
        weight_cfg={
            "selection": M.W_SELECT_PROB,
            "pct": config.get("weight_pct", 0.3),
            "mod": M.M_GAUSSIAN,
            "mod_param": config.get("weight_std", 0.05),
        },
        connection_cfg={
            "selection": M.C_SELECT_PCT_SHUFFLED,
            "pct": 0.0,
            "mod": M.M_GAUSSIAN,
            "mod_param": 0.3,
        },
        neuron_cfg={
            "selection": M.C_SELECT_PCT_SHUFFLED,
            "pct": 0.0,
            "mod": M.N_SPLIT_INCOMING_ONE,
        },
        pruning_cfg={
            "selection": M.P_SELECT_PCT_SHUFFLED,
            "pct": 0.0,
        },
    )
    pop = Population(
        n_inputs=4, n_outputs=2, size=config.get("pop_size", 80),
        init_conns_multiplier=config.get("init_mult", 2.0),
        init_neuron_range=(0, config.get("init_neurons", 1)),
        asexual_pct=0.8, crossover_pct=0.2,
        n_interspecies=1, n_elitism=config.get("elitism", 3),
        cull_pct=config.get("cull_pct", 0.6),
        optimizer=opt, mutation_policy=mut,
        speciation_policy="purge_then_standard",
        target_species=config.get("target_species", 5),
        threshold=config.get("threshold", 0.25),
        min_threshold=0.05, max_threshold=0.4, threshold_adjust=0.02,
        similarity_method="percentage",
        seed=config.get("seed", 0),
    )
    return pop


def evaluate_with_trace(genome: Genome, env_seed: int = 42, max_steps: int = 500) -> Dict:
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=env_seed)
    trace = {"obs": [], "actions": [], "reward": 0.0, "steps": 0}
    for _ in range(max_steps):
        a = act_discrete(genome, np.asarray(obs, dtype=np.float64))
        trace["obs"].append([float(x) for x in obs])
        trace["actions"].append(int(a))
        obs, r, terminated, truncated, _ = env.step(a)
        trace["reward"] += r
        trace["steps"] += 1
        if terminated or truncated:
            break
    env.close()
    return trace


def make_fitness_fn(env_seed: int = 42, max_steps: int = 500, n_avg: int = 1):
    def fit(genome: Genome) -> float:
        rs = []
        for k in range(n_avg):
            trace = evaluate_with_trace(genome, env_seed=env_seed + k, max_steps=max_steps)
            rs.append(trace["reward"])
            # capture traces of the best genome for visualization
            with STATE.lock:
                if STATE.population is not None and genome is STATE.population.best_genome:
                    STATE.episode_buffer.append({
                        "genome_id": id(genome),
                        "seed": env_seed + k,
                        "trace": trace,
                        "timestamp": time.time(),
                        "generation": STATE.population.generation,
                    })
                    if len(STATE.episode_buffer) > STATE.max_episode_buffer:
                        STATE.episode_buffer.pop(0)
        return float(np.mean(rs))
    return fit


# ----------------------------------------------------------------- training -
def training_loop(config: Dict) -> None:
    STATE.stop_flag.clear()
    try:
        pop = build_population(config)
        with STATE.lock:
            STATE.population = pop
            STATE.config = config
        fit_fn = make_fitness_fn(
            env_seed=config.get("env_seed", 42),
            max_steps=config.get("max_steps", 500),
            n_avg=config.get("n_avg", 1),
        )
        n_gens = config.get("generations", 100)
        for gen in range(n_gens):
            if STATE.stop_flag.is_set():
                break
            stats = pop.step(fit_fn)
            snap = pop.snapshot()
            with STATE.lock:
                STATE.latest_snapshot = snap
            if STATE.loop:
                try:
                    asyncio.run_coroutine_threadsafe(notify_subscribers(snap), STATE.loop)
                except Exception:
                    pass
            delay = STATE.generation_delay
            if delay > 0:
                time.sleep(delay)
    except Exception as e:
        import traceback
        print(f"Training loop error: {e}", flush=True)
        traceback.print_exc()
    finally:
        STATE.running = False
        if STATE.loop:
            try:
                asyncio.run_coroutine_threadsafe(notify_subscribers({"running": False}), STATE.loop)
            except Exception:
                pass


async def notify_subscribers(snap: Dict) -> None:
    for q in list(STATE.subscribers):
        try:
            q.put_nowait(snap)
        except asyncio.QueueFull:
            pass


# ----------------------------------------------------------------- FastAPI --
app = FastAPI(title="NEAT Visualizer Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "running": STATE.running}


@app.get("/api/state")
async def get_state():
    with STATE.lock:
        snap = dict(STATE.latest_snapshot) if STATE.latest_snapshot else {}
        snap["running"] = STATE.running
        snap["episode_buffer"] = list(STATE.episode_buffer[-5:])
        return snap


@app.post("/api/start")
async def start_training(config: Dict = Body(default_factory=dict)):
    if config is None:
        config = {}
    if STATE.running:
        return {"error": "already running"}
    STATE.running = True
    STATE.thread = threading.Thread(target=training_loop, args=(config,), daemon=True)
    STATE.thread.start()
    return {"status": "started", "config": config}


@app.post("/api/stop")
async def stop_training():
    STATE.stop_flag.set()
    return {"status": "stopping"}


@app.post("/api/set_delay")
async def set_delay(payload: Dict = Body(default_factory=dict)):
    payload = payload or {}
    STATE.generation_delay = float(payload.get("delay", 0.0))
    return {"delay": STATE.generation_delay}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    STATE.subscribers.append(q)
    if STATE.loop is None:
        STATE.loop = asyncio.get_event_loop()
    try:
        with STATE.lock:
            if STATE.latest_snapshot:
                await ws.send_json(STATE.latest_snapshot)
        while True:
            try:
                snap = await asyncio.wait_for(q.get(), timeout=10.0)
                await ws.send_json(snap)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "heartbeat", "running": STATE.running})
    except WebSocketDisconnect:
        pass
    finally:
        if q in STATE.subscribers:
            STATE.subscribers.remove(q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
