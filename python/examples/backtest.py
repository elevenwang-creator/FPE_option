import fpe_engine

result = fpe_engine.price({
    "kappa": 1.2,
    "theta": 0.05,
    "sigma": 0.35,
    "rho": -0.4,
    "r": 0.05,
    "T": 0.5,
    "S0": 100.0,
    "V0": 0.1,
    "K": [95.0, 100.0, 105.0],
    "barrier": 120.0,
    "option_type": "up_and_out_call",
    "n_s": 8,
    "n_v": 8,
    "rtol": 1e-4,
    "atol": 1e-6,
})

for i, K in enumerate([95.0, 100.0, 105.0]):
    print(f"K={K}: price={result['prices'][i]:.4f}, delta={result['deltas'][i]:.4f}")
