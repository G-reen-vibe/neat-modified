"""
Web-based visualizer for the modified NEAT implementation.

Features:
- Live training view with charts (fitness over time, species count, etc.)
- Genome graph visualization (network topology with weighted edges)
- Species panel (per-species members, best fitness, representative graph)
- Agent playback (renders the best genome in the env)
- Detailed stats panel (population-level metrics)

Run with:
    python visualizer/run.py --env CartPole-v1 --pop 100 --gens 100

Then open http://localhost:8000 in your browser.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import threading
import time
import base64
import io
from typing import Dict, List, Optional, Any
import numpy as np

# Make `neat` importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from neat import Config, Population, Genome
from neat.envs import make_env
from scripts.train import build_default_config


# ---------------------------------------------------------------------------
# Global state (shared between training thread and web server)
# ---------------------------------------------------------------------------
class VisState:
    def __init__(self):
        self.lock = threading.RLock()
        self.pop: Optional[Population] = None
        self.env_name: str = "CartPole-v1"
        self.running: bool = False
        self.paused: bool = False
        self.generation: int = 0
        self.history: List[Dict] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = -float("inf")
        self.episode_frames: List[str] = []   # base64 JPEGs for playback
        self.episode_reward: float = 0.0
        self.episode_steps: int = 0
        self.subscribers: List = []  # websocket connections
        self.config: Optional[Config] = None
        self.last_eval_summary: Dict = {}

    def broadcast(self, msg: Dict) -> None:
        """Send a message to all subscribed websockets."""
        dead = []
        for i, ws in enumerate(self.subscribers):
            try:
                asyncio.run_coroutine_threadsafe(
                    ws.send_json(msg), self.loop
                )
            except Exception:
                dead.append(i)
        for i in reversed(dead):
            self.subscribers.pop(i)


STATE = VisState()


# ---------------------------------------------------------------------------
# Training loop (runs in a background thread)
# ---------------------------------------------------------------------------
def training_thread(n_generations: int, max_steps: int, n_eval_episodes: int):
    STATE.running = True
    env_wrapper = make_env(STATE.env_name, max_steps=max_steps,
                           n_eval_episodes=n_eval_episodes,
                           seed=STATE.config.seed)
    while STATE.running and STATE.generation < n_generations:
        if STATE.paused:
            time.sleep(0.1)
            continue
        # Evaluate
        ep_seed = STATE.config.seed * 10000 + STATE.generation
        def eval_fn(g, _ep_seed=ep_seed):
            r, _, _ = env_wrapper.evaluate(g, episode_seed=_ep_seed)
            return r
        STATE.pop.evaluate(eval_fn)
        gen_best = STATE.pop.best()
        with STATE.lock:
            if gen_best and gen_best.fitness > STATE.best_fitness:
                STATE.best_fitness = gen_best.fitness
                STATE.best_genome = gen_best.clone()
        stats = STATE.pop.step()
        with STATE.lock:
            STATE.history.append(stats)
            STATE.generation = STATE.pop.generation
            STATE.last_eval_summary = {
                "max": stats["fitness_max"],
                "mean": stats["fitness_mean"],
                "min": stats["fitness_min"],
                "n_species": stats["n_species"],
                "avg_conns": stats["avg_conns"],
                "avg_nodes": stats["avg_nodes"],
            }
        # Broadcast update to subscribers
        STATE.broadcast({
            "type": "gen_update",
            "data": {
                "generation": STATE.generation,
                "stats": stats,
                "best_fitness": STATE.best_fitness,
                "species": [
                    {"id": s.id, "members": len(s.members),
                     "best_fitness": s.best_fitness}
                    for s in STATE.pop.speciator.species.values()
                ],
            }
        })
    STATE.running = False
    STATE.broadcast({"type": "done", "data": {"best_fitness": STATE.best_fitness}})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="NEAT Visualizer")


@app.get("/")
async def index():
    return HTMLResponse(HTML_PAGE)


@app.get("/api/state")
async def get_state():
    with STATE.lock:
        return JSONResponse({
            "running": STATE.running,
            "paused": STATE.paused,
            "generation": STATE.generation,
            "env_name": STATE.env_name,
            "best_fitness": STATE.best_fitness,
            "history": STATE.history,
            "last_eval_summary": STATE.last_eval_summary,
            "config": STATE.config.to_dict() if STATE.config else None,
            "species": [
                {"id": s.id, "members": len(s.members),
                 "best_fitness": s.best_fitness}
                for s in (STATE.pop.speciator.species.values() if STATE.pop else [])
            ],
        })


@app.get("/api/genome/{gid}")
async def get_genome(gid: int):
    """Return a genome's structure for visualization."""
    with STATE.lock:
        if not STATE.pop:
            return JSONResponse({"error": "no population"}, status_code=400)
        g = None
        for gg in STATE.pop.genomes:
            if gg.id == gid:
                g = gg
                break
        if g is None and STATE.best_genome:
            g = STATE.best_genome
        if g is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(_genome_to_vis(g))


def _genome_to_vis(g: Genome) -> Dict:
    """Convert a genome to a JSON-serializable dict for vis.js."""
    nodes = []
    for nid, n in g.nodes.items():
        nodes.append({
            "id": nid,
            "label": str(nid),
            "kind": n.kind,
            "activation": n.activation.kind,
        })
    edges = []
    for c in g.conns.values():
        edges.append({
            "id": c.innov,
            "from": c.src,
            "to": c.dst,
            "weight": c.weight,
            "width": min(5.0, abs(c.weight) * 2.0),
            "color": "#2ecc71" if c.weight > 0 else "#e74c3c",
        })
    return {
        "id": g.id,
        "fitness": g.fitness,
        "nodes": nodes,
        "edges": edges,
        "n_inputs": g.cfg.n_inputs,
        "n_outputs": g.cfg.n_outputs,
        "bias_id": g.bias_id,
    }


@app.get("/api/best_genomes")
async def best_genomes(limit: int = 10):
    """Top genomes by fitness in the current population."""
    with STATE.lock:
        if not STATE.pop:
            return JSONResponse([])
        sorted_g = sorted(STATE.pop.genomes, key=lambda g: g.fitness, reverse=True)
        return JSONResponse([{
            "id": g.id,
            "fitness": g.fitness,
            "n_conns": len(g.conns),
            "n_nodes": len(g.nodes),
            "species": g.parent_species,
        } for g in sorted_g[:limit]])


@app.post("/api/play_best")
async def play_best():
    """Run the best genome in the env and return frames."""
    if not STATE.best_genome:
        return JSONResponse({"error": "no best genome"}, status_code=400)
    env_wrapper = make_env(STATE.env_name, max_steps=500, n_eval_episodes=1,
                           seed=STATE.config.seed, render_mode="rgb_array")
    result = env_wrapper.rollout(STATE.best_genome, render=True)
    # Encode frames as base64 JPEG
    from PIL import Image
    frames_b64 = []
    for frame in result["frames"][:200]:  # cap at 200 frames
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode())
    with STATE.lock:
        STATE.episode_frames = frames_b64
        STATE.episode_reward = result["total_reward"]
        STATE.episode_steps = result["steps"]
    return JSONResponse({
        "n_frames": len(frames_b64),
        "reward": result["total_reward"],
        "steps": result["steps"],
    })


@app.get("/api/frames")
async def get_frames():
    with STATE.lock:
        return JSONResponse({
            "frames": STATE.episode_frames,
            "reward": STATE.episode_reward,
            "steps": STATE.episode_steps,
        })


@app.post("/api/pause")
async def pause():
    STATE.paused = not STATE.paused
    return JSONResponse({"paused": STATE.paused})


@app.post("/api/stop")
async def stop():
    STATE.running = False
    return JSONResponse({"stopped": True})


# ---------------------------------------------------------------------------
# WebSocket for live updates
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    STATE.subscribers.append(websocket)
    STATE.loop = asyncio.get_event_loop()
    try:
        # Send initial state
        with STATE.lock:
            await websocket.send_json({
                "type": "init",
                "data": {
                    "generation": STATE.generation,
                    "history": STATE.history,
                    "best_fitness": STATE.best_fitness,
                    "env_name": STATE.env_name,
                }
            })
        while True:
            await asyncio.sleep(1.0)
            # heartbeat
            await websocket.send_json({"type": "ping", "data": {}})
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in STATE.subscribers:
            STATE.subscribers.remove(websocket)


# ---------------------------------------------------------------------------
# HTML page (single-page app)
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>NEAT Visualizer</title>
  <meta charset="utf-8">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; }
    .header { background: #16213e; padding: 12px 24px; display: flex; justify-content: space-between; }
    .header h1 { font-size: 20px; color: #4cc9f0; }
    .header .controls button { background: #4361ee; color: white; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; margin-left: 8px; }
    .header .controls button:hover { background: #3a56d4; }
    .container { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; padding: 12px; }
    .panel { background: #16213e; border-radius: 8px; padding: 12px; min-height: 200px; }
    .panel h2 { color: #4cc9f0; font-size: 14px; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 1px; }
    .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    .stat { background: #0f3460; padding: 8px; border-radius: 4px; }
    .stat .label { font-size: 11px; color: #aaa; }
    .stat .value { font-size: 18px; font-weight: bold; color: #4cc9f0; }
    .chart-container { height: 180px; }
    .genome-list { max-height: 280px; overflow-y: auto; }
    .genome-item { background: #0f3460; padding: 8px; margin-bottom: 4px; border-radius: 4px; cursor: pointer; }
    .genome-item:hover { background: #1a4080; }
    .genome-item.selected { background: #4361ee; }
    .network-viz { height: 380px; background: #0f3460; border-radius: 4px; }
    .species-list { max-height: 200px; overflow-y: auto; }
    .species-item { background: #0f3460; padding: 6px; margin-bottom: 4px; border-radius: 4px; font-size: 12px; }
    .playback { background: #0f3460; border-radius: 4px; padding: 8px; text-align: center; }
    .playback img { max-width: 100%; border-radius: 4px; }
    .playback-controls button { background: #4361ee; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; margin: 4px; }
    .log { font-family: monospace; font-size: 11px; max-height: 100px; overflow-y: auto; color: #aaa; }
  </style>
</head>
<body>
<div class="header">
  <h1>NEAT Visualizer &mdash; <span id="env-name">...</span></h1>
  <div class="controls">
    <button onclick="togglePause()">Pause/Resume</button>
    <button onclick="playBest()">Play Best</button>
    <button onclick="refreshAll()">Refresh</button>
  </div>
</div>
<div class="container">
  <div class="panel">
    <h2>Stats</h2>
    <div class="stats-grid" id="stats"></div>
    <h2 style="margin-top:12px">Fitness Over Time</h2>
    <div class="chart-container"><canvas id="fitness-chart"></canvas></div>
    <h2 style="margin-top:12px">Species Count</h2>
    <div class="chart-container"><canvas id="species-chart"></canvas></div>
  </div>
  <div class="panel">
    <h2>Best Genome Network</h2>
    <div class="network-viz" id="network"></div>
    <h2 style="margin-top:12px">Genome List (Top 10)</h2>
    <div class="genome-list" id="genome-list"></div>
  </div>
  <div class="panel">
    <h2>Species</h2>
    <div class="species-list" id="species-list"></div>
    <h2 style="margin-top:12px">Playback</h2>
    <div class="playback">
      <img id="playback-img" src="" alt="No playback yet" />
      <div class="playback-controls">
        <button onclick="prevFrame()">Prev</button>
        <button onclick="playFrames()">Play</button>
        <button onclick="nextFrame()">Next</button>
      </div>
      <div id="playback-info" style="font-size:12px;color:#aaa;margin-top:4px"></div>
    </div>
    <h2 style="margin-top:12px">Log</h2>
    <div class="log" id="log"></div>
  </div>
</div>

<script>
let fitnessChart, speciesChart;
let networkInstance = null;
let currentGenomeId = null;
let frames = [];
let frameIdx = 0;
let playInterval = null;

function log(msg) {
  const el = document.getElementById('log');
  el.innerHTML = `[${new Date().toLocaleTimeString()}] ${msg}<br>` + el.innerHTML;
}

async function refreshAll() {
  await Promise.all([refreshState(), refreshGenomes(), refreshSpecies()]);
}

async function refreshState() {
  const r = await fetch('/api/state');
  const s = await r.json();
  document.getElementById('env-name').textContent = s.env_name;
  const stats = s.last_eval_summary || {};
  const html = `
    <div class="stat"><div class="label">Generation</div><div class="value">${s.generation}</div></div>
    <div class="stat"><div class="label">Best Fitness</div><div class="value">${s.best_fitness.toFixed(2)}</div></div>
    <div class="stat"><div class="label">Max (gen)</div><div class="value">${(stats.max||0).toFixed(2)}</div></div>
    <div class="stat"><div class="label">Mean (gen)</div><div class="value">${(stats.mean||0).toFixed(2)}</div></div>
    <div class="stat"><div class="label">Min (gen)</div><div class="value">${(stats.min||0).toFixed(2)}</div></div>
    <div class="stat"><div class="label">Species</div><div class="value">${stats.n_species||0}</div></div>
    <div class="stat"><div class="label">Avg Conns</div><div class="value">${(stats.avg_conns||0).toFixed(1)}</div></div>
    <div class="stat"><div class="label">Avg Nodes</div><div class="value">${(stats.avg_nodes||0).toFixed(1)}</div></div>
  `;
  document.getElementById('stats').innerHTML = html;
  updateCharts(s.history || []);
}

async function refreshGenomes() {
  const r = await fetch('/api/best_genomes?limit=10');
  const gs = await r.json();
  const el = document.getElementById('genome-list');
  el.innerHTML = gs.map(g => `
    <div class="genome-item ${g.id === currentGenomeId ? 'selected' : ''}"
         onclick="selectGenome(${g.id})">
      <div><strong>#${g.id}</strong> &mdash; fitness ${g.fitness.toFixed(2)}</div>
      <div style="font-size:11px;color:#aaa">
        conns=${g.n_conns} nodes=${g.n_nodes} species=${g.species}
      </div>
    </div>
  `).join('');
  // Auto-select the best one if nothing selected
  if (!currentGenomeId && gs.length > 0) {
    selectGenome(gs[0].id);
  }
}

async function selectGenome(gid) {
  currentGenomeId = gid;
  const r = await fetch(`/api/genome/${gid}`);
  const g = await r.json();
  drawNetwork(g);
  refreshGenomes();
}

function drawNetwork(g) {
  const nodes = g.nodes.map(n => {
    const colors = { input: '#4cc9f0', output: '#f72585', bias: '#ffd60a', hidden: '#4ade80' };
    return {
      id: n.id,
      label: n.label,
      color: colors[n.kind] || '#fff',
      shape: n.kind === 'input' || n.kind === 'output' || n.kind === 'bias' ? 'box' : 'dot',
      font: { color: '#000', size: 12 },
      x: n.kind === 'input' ? -200 : (n.kind === 'output' ? 200 : (Math.random() - 0.5) * 300),
      y: n.kind === 'input' ? (n.id * 60) : (n.kind === 'output' ? (n.id * 60) : (Math.random() - 0.5) * 300),
    };
  });
  const edges = g.edges.map(e => ({
    id: e.id,
    from: e.from,
    to: e.to,
    width: e.width,
    color: { color: e.color, opacity: 0.7 },
    arrows: 'to',
    title: `w=${e.weight.toFixed(3)}`,
  }));
  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  const options = {
    physics: { enabled: true, stabilization: { iterations: 50 } },
    nodes: { size: 16 },
    edges: { smooth: { type: 'continuous' } },
  };
  if (networkInstance) networkInstance.destroy();
  networkInstance = new vis.Network(document.getElementById('network'), data, options);
}

async function refreshSpecies() {
  const r = await fetch('/api/state');
  const s = await r.json();
  const el = document.getElementById('species-list');
  el.innerHTML = (s.species || []).map(sp => `
    <div class="species-item">
      <strong>Species ${sp.id}</strong> &mdash;
      ${sp.members} members, best=${sp.best_fitness.toFixed(2)}
    </div>
  `).join('');
}

function updateCharts(history) {
  const labels = history.map((_, i) => i + 1);
  const maxData = history.map(h => h.fitness_max);
  const meanData = history.map(h => h.fitness_mean);
  const minData = history.map(h => h.fitness_min);
  const speciesData = history.map(h => h.n_species);

  if (!fitnessChart) {
    fitnessChart = new Chart(document.getElementById('fitness-chart'), {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Max', data: [], borderColor: '#4cc9f0', backgroundColor: 'rgba(76,201,240,0.1)', tension: 0.3 },
        { label: 'Mean', data: [], borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.1)', tension: 0.3 },
        { label: 'Min', data: [], borderColor: '#f72585', backgroundColor: 'rgba(247,37,133,0.1)', tension: 0.3 },
      ]},
      options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false } } }
    });
  }
  fitnessChart.data.labels = labels;
  fitnessChart.data.datasets[0].data = maxData;
  fitnessChart.data.datasets[1].data = meanData;
  fitnessChart.data.datasets[2].data = minData;
  fitnessChart.update();

  if (!speciesChart) {
    speciesChart = new Chart(document.getElementById('species-chart'), {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Species', data: [], borderColor: '#ffd60a', backgroundColor: 'rgba(255,214,10,0.1)', tension: 0.3 }
      ]},
      options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false } } }
    });
  }
  speciesChart.data.labels = labels;
  speciesChart.data.datasets[0].data = speciesData;
  speciesChart.update();
}

async function togglePause() {
  await fetch('/api/pause', { method: 'POST' });
  log('Toggled pause');
}

async function playBest() {
  log('Playing best genome...');
  const r = await fetch('/api/play_best', { method: 'POST' });
  const result = await r.json();
  if (result.error) { log(result.error); return; }
  const fr = await fetch('/api/frames');
  const f = await fr.json();
  frames = f.frames;
  frameIdx = 0;
  showFrame();
  log(`Playback: reward=${f.reward.toFixed(2)}, steps=${f.steps}, frames=${f.frames.length}`);
}

function showFrame() {
  if (frames.length === 0) return;
  const img = document.getElementById('playback-img');
  img.src = 'data:image/jpeg;base64,' + frames[frameIdx];
  document.getElementById('playback-info').textContent =
    `Frame ${frameIdx+1}/${frames.length}`;
}

function nextFrame() {
  if (frameIdx < frames.length - 1) { frameIdx++; showFrame(); }
}
function prevFrame() {
  if (frameIdx > 0) { frameIdx--; showFrame(); }
}
function playFrames() {
  if (playInterval) { clearInterval(playInterval); playInterval = null; return; }
  playInterval = setInterval(() => {
    if (frameIdx >= frames.length - 1) {
      clearInterval(playInterval); playInterval = null; return;
    }
    nextFrame();
  }, 50);
}

// WebSocket for live updates
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'gen_update') {
    log(`Gen ${msg.data.generation}: max=${msg.data.stats.fitness_max.toFixed(2)} mean=${msg.data.stats.fitness_mean.toFixed(2)}`);
    refreshState();
    refreshSpecies();
    refreshGenomes();
  } else if (msg.type === 'done') {
    log(`Training done! Best fitness: ${msg.data.best_fitness.toFixed(2)}`);
  }
};

// Initial load
refreshAll();
setInterval(refreshAll, 3000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--gens", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--optimizer", action="store_true")
    p.add_argument("--no-train", action="store_true",
                   help="Don't start training; just serve the UI")
    args = p.parse_args()

    STATE.env_name = args.env
    STATE.config = build_default_config(args.env, seed=args.seed,
                                         pop_size=args.pop,
                                         optimizer=args.optimizer)
    STATE.pop = Population(STATE.config)

    if not args.no_train:
        t = threading.Thread(target=training_thread,
                             args=(args.gens, args.max_steps, args.eval_episodes),
                             daemon=True)
        t.start()

    print(f"\n=== NEAT Visualizer ===")
    print(f"  Env: {args.env}")
    print(f"  Pop: {args.pop}  Gens: {args.gens}")
    print(f"  Open http://localhost:{args.port} in your browser\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
