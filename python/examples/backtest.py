#!/usr/bin/env python3

import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    try:
        module = importlib.import_module("fpe_engine")
        is_available = module.is_available
        price_barrier_option = module.price_barrier_option
        solve_fpe = module.solve_fpe

        if not is_available():
            print("Mojo engine not available. Install with: pixi install")
            return

        print("Solving FPE for Heston params...")
        param_hash = solve_fpe(
            kappa=1.2,
            theta=0.05,
            sigma=0.35,
            rho=-0.4,
            r=0.05,
            T=0.5,
            S0=100.0,
            V0=0.1,
        )
        print(f"  param_hash = {param_hash}")

        print("Pricing barrier options...")
        for strike in [95.0, 100.0, 105.0]:
            result = price_barrier_option(
                S=100.0, K=strike, T=0.5, barrier=120.0, param_hash=param_hash
            )
            print(
                f"  K={strike}: price={result['price']:.4f}, delta={result['delta']:.4f}"
            )

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
