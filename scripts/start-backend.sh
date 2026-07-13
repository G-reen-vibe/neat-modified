#!/bin/bash
cd /home/z/my-project/neat_python
while true; do
  echo "[$(date)] Starting backend..."
  python3 visualizer/server.py --port 8000 2>&1
  echo "[$(date)] Backend exited with code $?, restarting in 3s..."
  sleep 3
done
