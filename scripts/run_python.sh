#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${1:-}" ]; then
    echo "Usage: scripts/run_python.sh <script.py> [args...]"
    echo "Available examples:"
    ls python/examples/*.py 2>/dev/null | xargs -I{} basename {} || echo "  (none found)"
    exit 1
fi

SCRIPT="$1"
shift

exec python "python/examples/${SCRIPT}" "$@"
