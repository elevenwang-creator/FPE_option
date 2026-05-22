"""FPE Option Pricing - Python backtest example.

Demonstrates pricing European and barrier options via the fpe_engine Python binding.
The binding uses a pre-compiled Mojo native module installed into site-packages.

Usage:
    pixi run py-run backtest.py
"""

import fpe_engine


def main():
    if not fpe_engine.is_available():
        print("ERROR: Mojo FPE engine not available.")
        print("Ensure Mojo SDK is installed and engine is built: pixi install && pixi run build")
        return

    print("=" * 60)
    print(" FPE Option Pricing - Python Binding")
    print("=" * 60)

    strikes = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]

    print("\n[1] European Call (option_type='european_call'):")
    result = fpe_engine.price(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.1, T=0.6, S0=60.0, V0=0.1,
        K=strikes,
        barrier=0.0,
        option_type="european_call",
        n_s=38, n_v=38,
    )
    for i, K in enumerate(strikes):
        print(f" K={K}: price={result['prices'][i]:.6f} success={result['success'][i]}")

    print("\n[2] Up-and-Out Call (barrier=80, option_type='up_and_out_call'):")
    result = fpe_engine.price(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.1, T=0.6, S0=60.0, V0=0.1,
        K=strikes,
        barrier=80.0,
        option_type="up_and_out_call",
        n_s=38, n_v=38,
    )
    for i, K in enumerate(strikes):
        print(f" K={K}: price={result['prices'][i]:.6f} success={result['success'][i]}")

    print("\n[3] Down-and-Out Call (barrier=50, option_type='down_and_out_call'):")
    result = fpe_engine.price(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.1, T=0.6, S0=60.0, V0=0.1,
        K=strikes,
        barrier=50.0,
        option_type="down_and_out_call",
        n_s=38, n_v=38,
    )
    for i, K in enumerate(strikes):
        print(f" K={K}: price={result['prices'][i]:.6f} success={result['success'][i]}")

    print("\n" + "=" * 60)
    print(" Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
