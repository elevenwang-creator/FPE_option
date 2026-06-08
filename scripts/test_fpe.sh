#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Find all Mojo test files, excluding NAIS, GPU, and slow feature tests
EXCLUDE_PATTERNS="test_nais_|test_gpu_|test_metal_gpu|test_cpu_gpu|test_cpu_parallel|test_max_integration|test_calibrator|test_bindings|test_e2e_pipeline|test_facade|test_fpe_engine|test_pricing_engine|test_four_pipelines|test_compute_pipeline|test_fbsde_tracked|test_critical_fixes|test_autograd_|test_adam|test_optim|test_newton_debug|test_root_cause|test_csc|test_schur_verify|test_sparse_lu_perf|test_new_operators|test_ddz_z3"
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
