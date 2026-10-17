#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/public/tbiom/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
CFG="${1:-configs/default.yaml}"
shift || true
exec "$PYTHON" -m lsfu_reid.report --config "$CFG" "$@"
