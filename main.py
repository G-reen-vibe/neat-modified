#!/usr/bin/env python3
"""
NEAT-Modified: Main Entrypoint

This script is the single entry point for the entire project. It can:
  1. Run the test suite
  2. Run a CartPole-v1 benchmark
  3. Record training data for the visualizer
  4. Start the visualizer backend (for live mode)
  5. Print project status

Usage:
    python3 main.py [command]

Commands:
    test        Run the pytest test suite
    benchmark   Run a CartPole-v1 benchmark
    record      Record training data for the visualizer (saves to public/training-data.json)
    serve       Start the FastAPI visualizer backend (live mode)
    status      Print project status
    help        Show this help

If no command is given, runs 'status'.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NEAT_PYTHON = os.path.join(PROJECT_ROOT, "neat_python")
WEBUI = os.path.join(PROJECT_ROOT, "webui")  # legacy
PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")


def cmd_test(args):
    """Run the pytest test suite."""
    print("=" * 60)
    print("Running test suite...")
    print("=" * 60)
    os.chdir(NEAT_PYTHON)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=NEAT_PYTHON,
    )
    return result.returncode


def cmd_benchmark(args):
    """Run a CartPole-v1 benchmark."""
    print("=" * 60)
    print("Running CartPole-v1 benchmark...")
    print("=" * 60)
    script = os.path.join(NEAT_PYTHON, "benchmarks", "run_cartpole.py")
    cmd = [sys.executable, script,
           "--gens", str(args.gens),
           "--pop", str(args.pop),
           "--seed", str(args.seed),
           "--n-avg", str(args.n_avg),
           "--out", args.out or os.path.join(PROJECT_ROOT, "download", "benchmark.json")]
    if args.no_optimizer:
        cmd.append("--no-optimizer")
    result = subprocess.run(cmd)
    return result.returncode


def cmd_record(args):
    """Record training data for the visualizer."""
    print("=" * 60)
    print("Recording training data for visualizer...")
    print("=" * 60)
    script = os.path.join(NEAT_PYTHON, "benchmarks", "record_training.py")
    result = subprocess.run([sys.executable, script])
    if result.returncode == 0:
        out = os.path.join(PUBLIC_DIR, "training-data.json")
        size_mb = os.path.getsize(out) / 1024 / 1024
        print(f"\n✓ Training data saved to {out} ({size_mb:.1f} MB)")
        print(f"  The visualizer will automatically use this data in playback mode.")
    return result.returncode


def cmd_serve(args):
    """Start the FastAPI visualizer backend."""
    print("=" * 60)
    print("Starting visualizer backend...")
    print("=" * 60)
    print(f"  Backend: http://localhost:{args.port}")
    print(f"  Health:  http://localhost:{args.port}/api/health")
    print(f"  State:   http://localhost:{args.port}/api/state")
    print(f"  WebSocket: ws://localhost:{args.port}/ws")
    print()
    print("  NOTE: The Z.ai sandbox may kill background processes.")
    print("  If the backend dies, the visualizer will fall back to")
    print("  playback mode using public/training-data.json.")
    print()
    script = os.path.join(NEAT_PYTHON, "visualizer", "server.py")
    cmd = [sys.executable, script, "--port", str(args.port)]
    result = subprocess.run(cmd)
    return result.returncode


def cmd_status(args):
    """Print project status."""
    print("=" * 60)
    print("NEAT-Modified Project Status")
    print("=" * 60)
    print()
    print("Project: Modified NEAT algorithm with GRPO-style optimizer,")
    print("         advanced speciation, and Next.js visualizer.")
    print("Repo:    https://github.com/G-reen-vibe/neat-modified")
    print()
    print("Structure:")
    print(f"  {NEAT_PYTHON}/")
    print("    neat/           - Core algorithm (11 modules)")
    print("    tests/          - 49 pytest tests")
    print("    benchmarks/     - CartPole-v1 benchmark + recording script")
    print("    visualizer/     - FastAPI backend + training worker")
    print(f"  {PROJECT_ROOT}/src/")
    print("    app/            - Next.js visualizer (page.tsx)")
    print("    components/neat/ - CartPole canvas, genome graph, species panel,")
    print("                      stats panel, control panel")
    print("    lib/            - neat-client.ts (WebSocket + playback)")
    print(f"  {PUBLIC_DIR}/")
    print("    training-data.json - 60-gen pre-recorded training data")
    print()

    # check tests
    print("Running quick test check...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=NEAT_PYTHON, capture_output=True, text=True,
    )
    last_line = result.stdout.strip().split("\n")[-1] if result.stdout else "no output"
    print(f"  Tests: {last_line}")
    print()

    # check training data
    td = os.path.join(PUBLIC_DIR, "training-data.json")
    if os.path.exists(td):
        size_mb = os.path.getsize(td) / 1024 / 1024
        print(f"  Training data: {td} ({size_mb:.1f} MB)")
    else:
        print(f"  Training data: NOT FOUND (run 'python3 main.py record' to generate)")
    print()

    print("To start the visualizer:")
    print("  1. The Next.js dev server runs automatically (port 3000)")
    print("  2. Open the preview URL in your browser")
    print("  3. Click 'Play Training' to watch the pre-recorded training")
    print()
    print("For live training (if backend can run):")
    print(f"  python3 {os.path.abspath(__file__)} serve")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="NEAT-Modified: Main Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # test
    p_test = sub.add_parser("test", help="Run the pytest test suite")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run a CartPole-v1 benchmark")
    p_bench.add_argument("--gens", type=int, default=30)
    p_bench.add_argument("--pop", type=int, default=50)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("--n-avg", type=int, default=1)
    p_bench.add_argument("--out", type=str, default=None)
    p_bench.add_argument("--no-optimizer", action="store_true")

    # record
    p_record = sub.add_parser("record", help="Record training data for the visualizer")

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI visualizer backend")
    p_serve.add_argument("--port", type=int, default=8000)

    # status
    p_status = sub.add_parser("status", help="Print project status")

    args = parser.parse_args()
    if not args.command:
        return cmd_status(args)

    commands = {
        "test": cmd_test,
        "benchmark": cmd_benchmark,
        "record": cmd_record,
        "serve": cmd_serve,
        "status": cmd_status,
    }
    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
