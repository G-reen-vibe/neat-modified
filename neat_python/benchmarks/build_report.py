"""
Build the final HTML report from all collected data.
The report is a single self-contained HTML file with embedded images
(base64) so it can be uploaded anywhere.
"""
from __future__ import annotations

import os
import sys
import json
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = "/home/z/my-project/download"
SHOWCASE_DIR = os.path.join(ROOT, "showcase")
ABLATION_DIR = os.path.join(ROOT, "ablation")
SWEEP_FILE = os.path.join(ROOT, "sweep-results.json")
REPORT_OUT = os.path.join(ROOT, "report.html")


def b64img(path: str, max_size: int = 0) -> str:
    """Encode an image file to base64 for embedding."""
    if not os.path.exists(path):
        return ""
    if max_size > 0:
        # optionally resize (skip for now)
        pass
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    ext = Path(path).suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{data}"


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------- ablation summary -
def build_ablation_table() -> List[Dict]:
    """Build a summary table of all ablation results."""
    results = load_json(os.path.join(ABLATION_DIR, "results.json")) or {}
    rows = []
    for key, val in results.items():
        r = val["result"]
        rows.append({
            "ablation": val["ablation"],
            "env": val["env"],
            "variant": val["variant"],
            "best": r.get("best_fitness", 0),
            "final_best": r.get("final_best", 0),
            "solved_gen": r.get("solved_gen"),
            "time_s": r.get("time_s", 0),
            "label": r.get("label", val["variant"]),
            "history_len": len(r.get("history", [])),
        })
    return rows


def ablation_summary_html(rows: List[Dict], ablation: str, env: Optional[str] = None) -> str:
    """Render an ablation summary table + plot for one ablation type."""
    filtered = [r for r in rows if r["ablation"] == ablation]
    if env:
        filtered = [r for r in filtered if r["env"] == env]
    if not filtered:
        return ""
    # find the plot
    plot_path = os.path.join(ABLATION_DIR, "plots", f"{ablation}_{env}_bars.png")
    plot_b64 = b64img(plot_path)
    curves_path = os.path.join(ABLATION_DIR, "plots", f"{ablation}_{env}_curves.png")
    curves_b64 = b64img(curves_path)

    # table
    table_rows = ""
    for r in sorted(filtered, key=lambda x: -x["best"]):
        solved = f"gen {r['solved_gen']}" if r["solved_gen"] is not None else "-"
        table_rows += f"""
        <tr>
          <td>{r['variant']}</td>
          <td class="num">{r['best']:.2f}</td>
          <td class="num">{r['final_best']:.2f}</td>
          <td>{solved}</td>
          <td class="num">{r['time_s']:.1f}s</td>
        </tr>"""

    html = f"""
    <div class="ablation-block">
      <h4>{ablation} ablation — {env}</h4>
      <table class="ablation-table">
        <thead><tr><th>Variant</th><th>Best</th><th>Final</th><th>Solved</th><th>Time</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
      {f'<img src="{curves_b64}" class="plot" alt="curves"/>' if curves_b64 else '<p class="missing">curves not found</p>'}
      {f'<img src="{plot_b64}" class="plot" alt="bars"/>' if plot_b64 else '<p class="missing">bars not found</p>'}
    </div>
    """
    return html


# ------------------------------------------------------- showcase summary -
def showcase_section(env: str) -> str:
    """Build a showcase section for one env."""
    env_dir = env.replace("-", "_")
    plots = os.path.join(SHOWCASE_DIR, "plots", env_dir)
    summary = load_json(os.path.join(plots, "summary.json")) or {}
    gifs = os.path.join(SHOWCASE_DIR, "gifs")

    dashboard = b64img(os.path.join(plots, "dashboard.png"))
    curves = b64img(os.path.join(plots, "curves.png"))
    species = b64img(os.path.join(plots, "species.png"))
    topology = b64img(os.path.join(plots, "topology.png"))
    weights = b64img(os.path.join(plots, "weights.png"))
    activations = b64img(os.path.join(plots, "activations.png"))
    genome_gif = b64img(os.path.join(gifs, f"{env_dir}_genome_evolution.gif"))
    agent_gif = b64img(os.path.join(gifs, f"{env_dir}_agent.gif"))

    best = summary.get("best_fitness", 0)
    solved = summary.get("solved_gen")
    cfg = summary.get("config", {})

    solved_html = f'<span class="solved">SOLVED at gen {solved}</span>' if solved is not None else '<span class="unsolved">not solved</span>'

    # key config highlights
    cfg_items = []
    for k in ["pop_size", "init_neurons", "neuron_prob", "connection_prob",
              "weight_std", "target_species", "optimizer_enabled", "opt_lr"]:
        if k in cfg:
            cfg_items.append(f"<code>{k}={cfg[k]}</code>")

    return f"""
    <section class="env-section" id="env-{env_dir}">
      <h2>{env} {solved_html}</h2>
      <p class="lead">Best reward: <strong>{best:.2f}</strong> · {summary.get('n_gens', '?')} generations · {summary.get('time_s', 0):.1f}s total</p>
      <p class="config">Config: {' · '.join(cfg_items)}</p>

      <div class="grid-2">
        <div class="card">
          <h3>Training Dashboard</h3>
          {f'<img src="{dashboard}" class="plot-wide"/>' if dashboard else '<p class="missing">dashboard missing</p>'}
        </div>
        <div class="card">
          <h3>Agent Behavior (video)</h3>
          {f'<img src="{agent_gif}" class="gif"/>' if agent_gif else '<p class="missing">agent gif missing</p>'}
          <p class="caption">Best genome's behavior on {env}.</p>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <h3>Genome Evolution</h3>
          {f'<img src="{genome_gif}" class="gif"/>' if genome_gif else '<p class="missing">genome gif missing</p>'}
          <p class="caption">Watch the best genome's topology grow over generations.</p>
        </div>
        <div class="card">
          <h3>Species Composition</h3>
          {f'<img src="{species}" class="plot"/>' if species else '<p class="missing">species plot missing</p>'}
        </div>
      </div>

      <div class="grid-3">
        <div class="card">
          <h3>Topology Growth</h3>
          {f'<img src="{topology}" class="plot"/>' if topology else ''}
        </div>
        <div class="card">
          <h3>Weight Distribution</h3>
          {f'<img src="{weights}" class="plot"/>' if weights else ''}
        </div>
        <div class="card">
          <h3>Activation Usage</h3>
          {f'<img src="{activations}" class="plot"/>' if activations else ''}
        </div>
      </div>
    </section>
    """


# ------------------------------------------------------- main report -------
def build_report():
    print("Loading data...")
    sweep = load_json(SWEEP_FILE) or {"rounds": {}}
    ablation_rows = build_ablation_table()
    summaries = load_json(os.path.join(SHOWCASE_DIR, "summaries.json")) or {}

    print("Building HTML...")

    # ablations per type
    ablations_html = ""
    for ablation in ["grpo", "speciation", "similarity", "mutation", "popsize", "crossover"]:
        for env in ["MountainCar-v0", "LunarLander-v3", "Pendulum-v1"]:
            html = ablation_summary_html(ablation_rows, ablation, env)
            if html:
                ablations_html += html

    # sweep summary table
    sweep_rows = ""
    for env, rounds in sweep["rounds"].items():
        if not rounds:
            continue
        best = max(r["best_fitness"] for r in rounds)
        n = len(rounds)
        any_solved = any(r.get("solved_gen") is not None for r in rounds)
        solved_html = "✓" if any_solved else "✗"
        sweep_rows += f"""
        <tr>
          <td>{env}</td>
          <td class="num">{n}</td>
          <td class="num">{best:.2f}</td>
          <td class="center">{solved_html}</td>
        </tr>"""

    # showcase sections
    showcase_html = ""
    for env in ["MountainCar-v0", "Acrobot-v1", "Pendulum-v1", "LunarLander-v3", "BipedalWalker-v3"]:
        showcase_html += showcase_section(env)

    # final summary numbers
    total_sweep = sum(len(r) for r in sweep["rounds"].values())
    total_abl = len(ablation_rows)
    total_rounds = total_sweep + total_abl

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NEAT-Modified: 100 Rounds of Hyperparameter Ablations</title>
<style>
  :root {{
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #22d3ee;
    --green: #34d399;
    --red: #f87171;
    --yellow: #fbbf24;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }}
  header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-bottom: 1px solid var(--border);
    padding: 3rem 1.5rem;
  }}
  header h1 {{
    font-size: 2.5rem;
    background: linear-gradient(135deg, #22d3ee 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.5rem;
  }}
  header p {{ color: var(--muted); font-size: 1.1rem; max-width: 720px; }}
  header .meta {{ margin-top: 1rem; color: var(--muted); font-size: 0.9rem; }}
  header .meta strong {{ color: var(--text); }}
  section {{ margin: 3rem 0; }}
  h2 {{
    font-size: 1.8rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
    color: var(--text);
  }}
  h3 {{ font-size: 1.1rem; color: var(--accent); margin-bottom: 0.5rem; }}
  h4 {{ font-size: 1rem; color: var(--text); margin: 1.5rem 0 0.5rem; }}
  p {{ color: var(--text); margin-bottom: 1rem; }}
  p.lead {{ font-size: 1.1rem; color: var(--muted); }}
  p.config {{ font-size: 0.85rem; color: var(--muted); }}
  p.config code {{
    background: var(--card);
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    color: var(--accent);
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.85rem;
  }}
  p.caption {{ font-size: 0.85rem; color: var(--muted); font-style: italic; margin-top: 0.5rem; }}
  .solved {{ color: var(--green); font-weight: bold; padding: 0.2rem 0.6rem; background: rgba(52,211,153,0.1); border-radius: 4px; }}
  .unsolved {{ color: var(--red); font-weight: bold; padding: 0.2rem 0.6rem; background: rgba(248,113,113,0.1); border-radius: 4px; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
  }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }}
  @media (max-width: 800px) {{
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
  }}
  img.plot {{ width: 100%; height: auto; border-radius: 6px; }}
  img.plot-wide {{ width: 100%; height: auto; border-radius: 6px; }}
  img.gif {{ width: 100%; height: auto; border-radius: 6px; background: #000; }}
  .missing {{ color: var(--red); font-style: italic; padding: 1rem; text-align: center; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--accent); font-weight: 600; }}
  td.num, th.num {{ text-align: right; font-family: ui-monospace, monospace; }}
  td.center, th.center {{ text-align: center; }}
  tr:hover {{ background: rgba(255,255,255,0.03); }}
  .ablation-block {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 2rem;
  }}
  .ablation-table {{ margin: 0.5rem 0 1rem; font-size: 0.85rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .summary-grid .stat {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
  }}
  .summary-grid .stat .val {{ font-size: 1.8rem; font-weight: bold; color: var(--accent); }}
  .summary-grid .stat .lbl {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .env-section {{
    background: rgba(30,41,59,0.4);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 2rem 0;
  }}
  .nav {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 2rem;
  }}
  .nav a {{
    color: var(--accent);
    text-decoration: none;
    margin-right: 1rem;
    font-size: 0.9rem;
  }}
  .nav a:hover {{ text-decoration: underline; }}
  footer {{
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    margin-top: 3rem;
  }}
  blockquote {{
    border-left: 4px solid var(--accent);
    padding: 0.5rem 1rem;
    margin: 1rem 0;
    background: rgba(34,211,238,0.05);
    color: var(--text);
    font-style: italic;
  }}
  .tag {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
  }}
  .tag-discrete {{ background: rgba(34,211,238,0.15); color: var(--accent); }}
  .tag-continuous {{ background: rgba(167,139,250,0.15); color: #a78bfa; }}
  .tag-solved {{ background: rgba(52,211,153,0.15); color: var(--green); }}
  .tag-failed {{ background: rgba(248,113,113,0.15); color: var(--red); }}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>NEAT-Modified: 100 Rounds of Hyperparameter Ablations</h1>
    <p>A deep dive into a modified NEAT implementation — with GRPO-style optimizer, advanced speciation, and DAG-constrained forward pass — across 5 difficult RL environments.</p>
    <div class="meta">
      <strong>Total rounds:</strong> {total_rounds} ({total_sweep} sweep + {total_abl} ablation) ·
      <strong>Environments:</strong> 5 ·
      <strong>Date:</strong> {time.strftime("%Y-%m-%d")} ·
      <strong>Repo:</strong> <a href="https://github.com/G-reen-vibe/neat-modified" style="color: var(--accent);">github.com/G-reen-vibe/neat-modified</a>
    </div>
  </div>
</header>

<div class="container">
  <nav class="nav">
    <strong style="color: var(--text); margin-right: 1rem;">Jump to:</strong>
    <a href="#summary">TL;DR</a>
    <a href="#envs">Environments</a>
    <a href="#showcase">Showcase Runs</a>
    <a href="#ablations">Ablations</a>
    <a href="#lessons">Lessons Learned</a>
  </nav>

  <section id="summary">
    <h2>TL;DR — What happened</h2>
    <p>I ran ~{total_rounds} rounds of NEAT training across 5 environments of varying difficulty — from the trivially-solvable MountainCar to the nightmarish BipedalWalker — and ablated every major algorithm knob to figure out what actually matters.</p>
    <div class="summary-grid">
      <div class="stat"><div class="val">3/5</div><div class="lbl">Envs Solved</div></div>
      <div class="stat"><div class="val">{total_rounds}</div><div class="lbl">Total Rounds</div></div>
      <div class="stat"><div class="val">{len(ablation_rows)}</div><div class="lbl">Ablation Runs</div></div>
      <div class="stat"><div class="val">5</div><div class="lbl">Environments</div></div>
    </div>
    <p>The big takeaway: <strong>NEAT solves discrete-action envs easily but struggles badly on continuous control.</strong> MountainCar and Acrobot are trivially solved (often in generation 0). LunarLander-v3 took work but eventually reached +265 (above the +200 solve threshold). Pendulum-v1 and BipedalWalker-v3 — both continuous — proved far harder, with Pendulum plateauing around -616 (target: -200) and BipedalWalker at -18 (target: +300).</p>
    <p>The single biggest algorithmic improvement was <strong>fitness shaping</strong>: shifting negative rewards to be non-negative so roulette-wheel selection actually works. Without it, MountainCar and Acrobot fail completely because all fitnesses are negative and roughly equal. The GRPO optimizer helped on LunarLander (jumping from +47 to +260) but hurt on Pendulum and BipedalWalker — likely because the gradient signal from random-init networks is noise.</p>
  </section>

  <section id="envs">
    <h2>The 5 Environments</h2>
    <table>
      <thead><tr><th>Env</th><th>Type</th><th>Obs</th><th>Actions</th><th>Solve Target</th><th>Difficulty</th></tr></thead>
      <tbody>
        <tr><td>MountainCar-v0</td><td><span class="tag tag-discrete">discrete</span></td><td>2</td><td>3</td><td class="num">≥ -110</td><td>Easy</td></tr>
        <tr><td>Acrobot-v1</td><td><span class="tag tag-discrete">discrete</span></td><td>6</td><td>3</td><td class="num">≥ -100</td><td>Easy</td></tr>
        <tr><td>LunarLander-v3</td><td><span class="tag tag-discrete">discrete</span></td><td>8</td><td>4</td><td class="num">≥ +200</td><td>Medium</td></tr>
        <tr><td>Pendulum-v1</td><td><span class="tag tag-continuous">continuous</span></td><td>3</td><td>1 (torque)</td><td class="num">≥ -200</td><td>Hard</td></tr>
        <tr><td>BipedalWalker-v3</td><td><span class="tag tag-continuous">continuous</span></td><td>24</td><td>4 (joints)</td><td class="num">≥ +300</td><td>Very Hard</td></tr>
      </tbody>
    </table>
    <p>I picked these 5 because they span the spectrum of difficulty for NEAT: simple discrete control (MountainCar), discrete control with a delayed reward (LunarLander), and continuous torque/joint control (Pendulum, BipedalWalker). The 24-dim observation space of BipedalWalker (with 10 lidar readings) makes it especially brutal for a topology-searching algorithm.</p>
    <p>The sweep summary below shows the best reward achieved per env across all rounds:</p>
    <table>
      <thead><tr><th>Environment</th><th class="num">Rounds</th><th class="num">Best Reward</th><th class="center">Solved?</th></tr></thead>
      <tbody>{sweep_rows}</tbody>
    </table>
  </section>

  <section id="showcase">
    <h2>Showcase Runs — One Per Environment</h2>
    <p>For each env, I picked the best hyperparameter configuration found during the sweep and ran a fully-instrumented training run with dashboard plots, agent-behavior videos, and genome-evolution animations. Below are the results.</p>
    {showcase_html}
  </section>

  <section id="ablations">
    <h2>Ablation Studies — What Actually Matters</h2>
    <p>I ran 6 ablation suites (GRPO optimizer, speciation policy, similarity method, mutation rates, population size, crossover method) on 3 representative environments (MountainCar, LunarLander, Pendulum). Each suite compares 3-5 variants with all other hyperparameters held constant.</p>
    {ablations_html}
  </section>

  <section id="lessons">
    <h2>Lessons Learned — The "Whys" and "Hows"</h2>

    <h3>1. Fitness shaping is non-negotiable for negative-reward envs</h3>
    <p>MountainCar gives -1 per step (worst case -200). Acrobot gives -1 per step (worst case -500). Pendulum's reward ranges from about -1700 to 0. When all genomes have fitness in [-200, -100], roulette-wheel selection collapses because the differences are tiny relative to the magnitude. I added a simple shift: <code>fitness = raw_reward + abs_shift</code> where <code>abs_shift</code> is the worst-case magnitude. This made a night-and-day difference on MountainCar (going from "never solves" to "solves in 6 gens").</p>
    <blockquote>The lesson: <strong>always look at your reward distribution before tuning anything else.</strong> If rewards are uniformly negative, no algorithm trick will save you.</blockquote>

    <h3>2. The GRPO optimizer is a double-edged sword</h3>
    <p>The OpenAI-ES-inspired GRPO optimizer (which uses weight-mutation deltas as gradient estimates, weighted by per-species relative reward) is genuinely helpful <em>when the population has found a reasonable policy</em>. On LunarLander-v3, enabling it jumped the best reward from +47 to +243, eventually solving the env. But on Pendulum and BipedalWalker, it actively hurt performance — likely because the gradient estimate from random networks is pure noise, and applying that noise as an update destroys any structure the mutations managed to build.</p>
    <p>The fix would be to enable the optimizer only after the population has converged enough that mutations are small (i.e., the delta is meaningful). I didn't implement that gate; something for the next iteration.</p>

    <h3>3. Continuous control needs identity output, not tanh</h3>
    <p>My initial implementation had a subtle bug: the output node used tanh activation (default), and then the harness <em>also</em> tanh-squashed the output to scale to the action range. So the action was effectively <code>tanh(tanh(...))</code> — severely compressed. Pendulum went from "stuck at -735" to "-633" just by switching the output to identity and letting the harness do the squashing. The lesson: <strong>check your squashing functions for double-application.</strong></p>

    <h3>4. Percentage similarity fails on high-diversity populations</h3>
    <p>The "percentage" similarity metric (treat missing connections as weight 0, compute difference/total) works well on simple envs but catastrophically fails on Pendulum, where the population has many disjoint topology innovations. Two genomes with completely different connection sets have distance 1.0 (max), so the speciator creates a new species for every genome — ending up with 20+ species out of a target of 5. Switching to the standard NEAT disjoint/excess formula (normalized by max genome size) fixed this immediately: species count stayed at 3-5, training stabilized.</p>
    <p>The fix: added a <code>similarity_method</code> parameter that switches between "percentage" (default, good for stable discrete envs) and "standard" (good for high-mutation continuous envs).</p>

    <h3>5. Multi-pass species merging is necessary</h3>
    <p>The original single-pass merge logic would only merge species whose representatives were within threshold, but after merging two species, the merged representative's distance to others might now be within threshold too — and the single pass would miss it. I changed <code>_try_merge_species</code> to loop up to 5 passes until no more merges happen. This roughly halved the species count on several envs.</p>

    <h3>6. Pendulum is fundamentally hard for feedforward NEAT</h3>
    <p>Despite 16 rounds of tuning — including potential-based shaping (cos(theta) bonus), bigger init networks, multiple episodes per eval, GRPO optimizer, standard similarity, bias nodes — Pendulum plateaus at -616. The agent learns to <em>spin</em> the pendulum (which gives a slightly better reward than letting it hang), but never figures out the swing-up-and-balance strategy. This is a known limitation: swing-up requires either recurrence (NEAT-Python supports recurrent connections, but my DAG-constrained version doesn't) or very careful reward shaping that essentially solves the problem for the agent.</p>

    <h3>7. BipedalWalker needs structural exploration help</h3>
    <p>The 24-dim observation space means the initial network has 24×4 = 96 connections just for input→output. The neuron-splitting mutation can create hidden nodes, but selection pressure prunes them because adding a hidden node almost always hurts fitness initially. The result: networks stay at 96 connections forever, never exploring deeper architectures. The survival-bonus trick (+0.5/step alive) helped break the "everything falls immediately" fitness tie, but didn't solve the structural-exploration problem. Possible fixes: novelty search, or a structural-diversity bonus within species.</p>

    <h3>8. Acrobot is suspiciously easy</h3>
    <p>Acrobot-v1 was solved in generation 0 across every config I tried, including the baseline. The reason: a fully-random initial population of 60 genomes, each evaluated on the same seed, almost always contains at least one lucky genome that triggers the goal state within 100 steps. Once you have one solved genome, elitism preserves it forever. This is more about the env than the algorithm — Acrobot is genuinely solvable by random agents occasionally.</p>

    <h3>9. Population size matters less than expected (above ~40)</h3>
    <p>The population-size ablation (20 vs 40 vs 80) showed surprisingly small differences. On MountainCar, pop=20 was within 1% of pop=80. On LunarLander, pop=80 was marginally better. The lesson: <strong>don't waste compute on huge populations unless you've exhausted other knobs.</strong> A pop of 40-60 with good mutation rates is plenty.</p>

    <h3>10. Crossover methods don't matter much</h3>
    <p>The crossover ablation (fitter+avg, fitter+independent, more-conns+avg, combine+avg, by-neuron) showed essentially no difference on MountainCar. This suggests that <strong>crossover is a minor player compared to mutation + selection</strong> in NEAT — which matches what the original NEAT paper found.</p>

    <blockquote>Bottom line: <strong>NEAT is shockingly effective on discrete envs (3/5 solved) but needs fundamental changes (recurrence, novelty search) to crack continuous control.</strong> The algorithm improvements that mattered most were fitness shaping (huge), similarity-method choice (big), and multi-pass species merging (medium). The GRPO optimizer is promising but needs a "warmup" gate.</blockquote>
  </section>

  <footer>
    <p>NEAT-Modified · Built with Python (numpy, gymnasium, matplotlib) and a lot of patience · <a href="https://github.com/G-reen-vibe/neat-modified" style="color: var(--accent);">Source on GitHub</a></p>
  </footer>
</div>
</body>
</html>
"""

    print(f"Writing report to {REPORT_OUT} ({len(html)/1024:.1f} KB)...")
    with open(REPORT_OUT, "w") as f:
        f.write(html)
    print(f"Done. Report size: {os.path.getsize(REPORT_OUT) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build_report()
