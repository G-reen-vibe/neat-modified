#!/bin/bash
# Watchdog: restart the NEAT backend if it dies
# Usage: nohup bash watchdog.sh &

BACKEND_CMD="python3 /home/z/my-project/neat_python/visualizer/server.py --port 8000"
LOG_FILE="/home/z/my-project/.zscripts/neat-backend.log"
PID_FILE="/home/z/my-project/.zscripts/neat-backend.pid"

while true; do
  if ! pgrep -f "server.py --port 8000" > /dev/null; then
    echo "[$(date)] Backend down, restarting..." >> "$LOG_FILE"
    cd /home/z/my-project/neat_python
    nohup $BACKEND_CMD >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 3
    if pgrep -f "server.py --port 8000" > /dev/null; then
      echo "[$(date)] Backend restarted (PID $(cat $PID_FILE))" >> "$LOG_FILE"
    else
      echo "[$(date)] Backend failed to start" >> "$LOG_FILE"
    fi
  fi
  sleep 10
done
