#!/usr/bin/env bash
set -u

max_restarts=50
restart_count=0

while true; do
    echo "[$(date '+%F %T')] Starting training (restart ${restart_count}/${max_restarts})"
    python -u train.py
    status=$?

    if [ "$status" -eq 0 ]; then
        echo "[$(date '+%F %T')] Training completed successfully"
        exit 0
    fi

    restart_count=$((restart_count + 1))
    if [ "$restart_count" -gt "$max_restarts" ]; then
        echo "[$(date '+%F %T')] Training stopped after ${max_restarts} automatic restarts"
        exit "$status"
    fi

    echo "[$(date '+%F %T')] Training exited with status ${status}; retrying in 10 seconds"
    sleep 10
done
