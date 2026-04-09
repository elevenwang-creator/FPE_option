from std.sys import has_accelerator
from engines.fpe.gpu.executor import GPUFullChainExecutor


def validate_batch_size[batch_size: Int]() raises:
    print("Validating Batch Size:", batch_size)
    var executor = GPUFullChainExecutor[batch_size](n_s=8, n_v=8)

    # Execute pricing chain
    executor.execute_batch_pricing()
    print("  [OK] Pricing chain executed.")

    # Execute calibration chain
    # executor.execute_calibration_logic()
    print("  [OK] Calibration chain (skipped).")


def main() raises:
    if not has_accelerator():
        print("GPU not found.")
        return

    print("=== Heston GPU Logic Chain Validation ===")

    validate_batch_size[4]()
    validate_batch_size[16]()

    print("Validation complete!")
