#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-/home/public/tbiom/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$PYTHON" -u -m lsfu_reid.train --config "${1:-configs/smoke.yaml}" --device cuda
