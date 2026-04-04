#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize the FPE pricing engine. Returns 0 on success. */
int32_t fpe_init(void);

/** Cleanup resources. */
void fpe_destroy(void);

/**
 * Price a single option (CPU, sub-millisecond).
 * @param S     Current stock price
 * @param K     Strike price
 * @param V     Current variance
 * @param barrier  Barrier level
 * @param payoff_type  0=BarrierUpAndOut, 1=EuropeanCall
 * @param param_hash  Hash of Heston parameters (from fpe_solve_fpe)
 * @param out_price  Output: option price
 * @param out_delta  Output: delta
 * @param out_gamma  Output: gamma
 * @return 0 on success, non-zero on error
 */
int32_t fpe_price_single(
    double S,
    double K,
    double V,
    double barrier,
    int32_t payoff_type,
    uint64_t param_hash,
    double* out_price,
    double* out_delta,
    double* out_gamma
);

/**
 * Price a batch of options (GPU, high throughput).
 * @param count Number of options
 * @return 0 on success, non-zero on error
 */
int32_t fpe_price_batch(
    const double* S,
    const double* K,
    const double* T,
    const double* barrier,
    const int32_t* payoff_type,
    int32_t count,
    uint64_t param_hash,
    double* out_prices,
    double* out_deltas,
    double* out_gammas
);

#ifdef __cplusplus
}
#endif
