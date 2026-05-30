#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PREFIX="${PIXI_ENV_PREFIX:-.pixi/envs/default}"

if [ -z "${1:-}" ]; then
    echo "Usage: scripts/run_cpp.sh <example_name>"
    echo "Available examples:"
    ls cpp/examples/build/demo 2>/dev/null && echo "  demo" || echo "  (none built — run: pixi run build)"
    exit 1
fi

EXAMPLE="cpp/examples/build/$1"
if [ ! -f "${EXAMPLE}" ]; then
    echo "Error: ${EXAMPLE} not found. Run 'pixi run build' first."
    exit 1
fi

export DYLD_LIBRARY_PATH="${PREFIX}/lib:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}"
exec "${EXAMPLE}"
