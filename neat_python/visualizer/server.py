"""
NEAT Visualizer Backend (thin).

Reads training state from a file written by training_worker.py.
Starts/stops the worker process on demand.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


STATE_FILE = "/home/z/my-project/.zscripts/training-state.json"
CONFIG_FILE = "/home/z/my-project/.zscripts/training-config.json"
CONTROL_FILE = "/home/z/my-project/.zscripts/training-control.json"
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "training_worker.py")
WORKER_LOG = "/home/z/my-project/.zscripts/training-worker.log"


class ServerState:
    def __init__(self) -> None:
        self.worker_process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.subscribers: List[asyncio.Queue] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.poll_thread: Optional[threading.Thread] = None
        self.stop_poll = threading.Event()


STATE = ServerState()


def read_state_file() -> Dict:
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"running": False, "message": "no training yet"}


def write_control(ctrl: Dict) -> None:
    with open(CONTROL_FILE, "w") as f:
        json.dump(ctrl, f)


def write_config(cfg: Dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)


def start_worker(config: Dict) -> bool:
    with STATE.lock:
        if STATE.worker_process is not None and STATE.worker_process.poll() is None:
            return False
        write_config(config)
        log_f = open(WORKER_LOG, "a")
        STATE.worker_process = subprocess.Popen(
            [sys.executable, WORKER_SCRIPT, "--config", CONFIG_FILE],
            stdout=log_f, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(WORKER_SCRIPT),
        )
    # start polling thread if not running
    if STATE.poll_thread is None or not STATE.poll_thread.is_alive():
        STATE.stop_poll.clear()
        STATE.poll_thread = threading.Thread(target=poll_state_loop, daemon=True)
        STATE.poll_thread.start()
    return True


def stop_worker() -> None:
    write_control({"stop": True})
    # also try sending SIGTERM
    with STATE.lock:
        if STATE.worker_process is not None and STATE.worker_process.poll() is None:
            try:
                STATE.worker_process.terminate()
            except Exception:
                pass


def poll_state_loop() -> None:
    """Continuously read the state file and notify subscribers."""
    last_mtime = 0
    while not STATE.stop_poll.is_set():
        try:
            mtime = os.path.getmtime(STATE_FILE)
            if mtime != last_mtime:
                last_mtime = mtime
                state = read_state_file()
                # check if worker is still alive
                with STATE.lock:
                    worker_alive = (STATE.worker_process is not None
                                     and STATE.worker_process.poll() is None)
                state["running"] = worker_alive and state.get("running", False)
                if STATE.loop:
                    try:
                        asyncio.run_coroutine_threadsafe(notify_subscribers(state), STATE.loop)
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(0.3)


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
    with STATE.lock:
        worker_alive = (STATE.worker_process is not None
                         and STATE.worker_process.poll() is None)
    return {"status": "ok", "running": worker_alive}


@app.get("/api/state")
async def get_state():
    state = read_state_file()
    with STATE.lock:
        worker_alive = (STATE.worker_process is not None
                         and STATE.worker_process.poll() is None)
    state["running"] = worker_alive and state.get("running", False)
    return state


@app.post("/api/start")
async def start_training(config: Dict = Body(default_factory=dict)):
    if config is None:
        config = {}
    with STATE.lock:
        if STATE.worker_process is not None and STATE.worker_process.poll() is None:
            return {"error": "already running"}
    ok = start_worker(config)
    if not ok:
        return {"error": "failed to start"}
    # start polling thread
    if STATE.loop is None:
        STATE.loop = asyncio.get_event_loop()
    if STATE.poll_thread is None or not STATE.poll_thread.is_alive():
        STATE.stop_poll.clear()
        STATE.poll_thread = threading.Thread(target=poll_state_loop, daemon=True)
        STATE.poll_thread.start()
    return {"status": "started", "config": config}


@app.post("/api/stop")
async def stop_training():
    stop_worker()
    return {"status": "stopping"}


@app.post("/api/set_delay")
async def set_delay(payload: Dict = Body(default_factory=dict)):
    payload = payload or {}
    delay = float(payload.get("delay", 0.0))
    write_control({"delay": delay})
    return {"delay": delay}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    STATE.subscribers.append(q)
    try:
        if STATE.loop is None:
            try:
                STATE.loop = asyncio.get_running_loop()
            except RuntimeError:
                STATE.loop = asyncio.get_event_loop()
        # send current state immediately
        state = read_state_file()
        try:
            import json as _json
            raw = _json.dumps(state, default=str)
            await ws.send_text(raw)
        except Exception:
            pass
        while True:
            try:
                snap = await asyncio.wait_for(q.get(), timeout=10.0)
                try:
                    import json as _json
                    raw = _json.dumps(snap, default=str)
                    await ws.send_text(raw)
                except Exception:
                    break
            except asyncio.TimeoutError:
                try:
                    await ws.send_json({"type": "heartbeat", "running": False})
                except Exception:
                    break
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}", flush=True)
    finally:
        if q in STATE.subscribers:
            try:
                STATE.subscribers.remove(q)
            except ValueError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
