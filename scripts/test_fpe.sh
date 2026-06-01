#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Find all Mojo test files, excluding NAIS and GPU tests
EXCLUDE_PATTERNS="test_nais_|test_gpu_|test_metal_gpu|test_cpu_gpu|test_max_integration"
TESTS=()
while IFS= read -r f; do
    TESTS+=("$f")
done < <(find tests -name '*.mojo' | grep -vE "$EXCLUDE_PATTERNS" | sort)

echo "=== Running ${#TESTS[@]} FPE Mojo tests ==="
for t in "${TESTS[@]}"; do
    echo "  → $(basename "$t")"
    pixi run mojo run -I src "$t"
done
echo "=== All ${#TESTS[@]} FPE tests passed ==="
