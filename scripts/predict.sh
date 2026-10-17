#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/public/tbiom/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CFG="${1:-configs/default.yaml}"
shift || true
exec "$PYTHON" -m lsfu_reid.predict --config "$CFG" "$@"
