/** @file fpe_engine.h
 *  @brief C ABI for the FPE Option Pricing Engine.
 *
 *  Provides a C interface to the Mojo-based Fokker-Planck equation solver
 *  for pricing European and single-barrier options under the Heston model.
 */

#ifndef FPE_ENGINE_H
#define FPE_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to an FPE compute pipeline. */
typedef struct FpeCompute FpeCompute;

/** Result: a 1-D array (e.g. initial condition coefficients). */
struct FpeVecResult {
    double* data;
    int32_t len;
};

/** Result: two 1-D arrays (e.g. s and v knot vectors). */
struct FpeVec2Result {
    double* s_data;
    int32_t s_len;
    double* v_data;
    int32_t v_len;
};

/** Result: grid points and integration weights in s and v. */
struct FpeGridPtsResult {
    double* s_data;
    int32_t s_len;
    double* v_data;
    int32_t v_len;
    double* sw_data;
    double* vw_data;
};

/** Result: a 2-D matrix (row-major). */
struct FpeMatResult {
    double* data;
    int32_t n_rows;
    int32_t n_cols;
};

/** Result: delta, gamma, vega for each strike. */
struct FpeGreeksResult {
    double* delta;
    double* gamma;
    double* vega;
    int32_t len;
};

/** Result: price + Greeks from the one-shot API. */
struct FpeOneshotResult {
    double* price;
    double* delta;
    double* gamma;
    double* vega;
    int32_t len;
};

/** Create an FPE compute pipeline (lazy evaluation).
 *
 *  @param kappa   Mean-reversion speed of variance
 *  @param theta   Long-term variance level
 *  @param sigma   Volatility of variance (eta)
 *  @param rho     Correlation between asset and variance processes
 *  @param r       Risk-free rate
 *  @param T       Time to maturity (years)
 *  @param S0      Initial asset price
 *  @param V0      Initial variance
 *  @param n_s     Number of B-spline knots in s-direction
 *  @param n_v     Number of B-spline knots in v-direction
 *  @param barrier Barrier level (0.0 = no barrier)
 *  @param option_type Option type (0-9, see README)
 *  @param num_insert   Number of quadrature grid points per direction
 *  @param s_min   Lower domain bound (< 0 → 0.0)
 *  @param s_max   Upper domain bound (≤ 0 → S0 * 3)
 *  @return Pipeline handle, or NULL on failure
 */
extern FpeCompute* fpe_compute_create(
    double kappa, double theta, double sigma, double rho,
    double r, double T, double S0, double V0,
    int32_t n_s, int32_t n_v,
    double barrier, int32_t option_type,
    int32_t num_insert,
    double s_min, double s_max
);

/** Destroy a pipeline handle created by fpe_compute_create(). */
extern void fpe_compute_destroy(FpeCompute* ptr);

/** Retrieve the B-spline knot vectors.
 *  @param ptr    Pipeline handle
 *  @param result Filled with s-knots and v-knots arrays
 */
extern void fpe_compute_knots(FpeCompute* ptr, struct FpeVec2Result* result);

/** Retrieve the quadrature grid points and weights.
 *  @param ptr    Pipeline handle
 *  @param result Filled with s/v grid points and integration weights
 */
extern void fpe_compute_grid_points(FpeCompute* ptr, struct FpeGridPtsResult* result);

/** Retrieve the initial condition coefficients.
 *  @param ptr    Pipeline handle
 *  @param result Filled with B-spline coefficients at t=0
 */
extern void fpe_compute_initial_condition(FpeCompute* ptr, struct FpeVecResult* result);

/** Run the ODE solve and retrieve all time steps.
 *  @param ptr    Pipeline handle
 *  @param result Filled with the full solution (time steps × degrees of freedom)
 */
extern void fpe_compute_solve(FpeCompute* ptr, struct FpeMatResult* result);

/** Retrieve the terminal PDF.
 *  @param ptr    Pipeline handle
 *  @param result Filled with the PDF evaluated at the quadrature grid
 */
extern void fpe_compute_pdf(FpeCompute* ptr, struct FpeMatResult* result);

/** Price one or more strikes using the computed terminal PDF.
 *  @param ptr    Pipeline handle
 *  @param K      Array of strike prices
 *  @param n_K    Number of strikes
 *  @param result Filled with prices (one per strike)
 */
extern void fpe_compute_price(
    FpeCompute* ptr,
    const double* K,
    int32_t n_K,
    struct FpeVecResult* result
);

/** Compute Greeks (delta, gamma, vega) for each strike via finite differences.
 *  @param ptr    Pipeline handle
 *  @param K      Array of strike prices
 *  @param n_K    Number of strikes
 *  @param rel_s  Relative perturbation for finite-difference in s (0.0 → default)
 *  @param rel_v  Relative perturbation for finite-difference in v (0.0 → default)
 *  @param result Filled with delta, gamma, vega arrays
 */
extern void fpe_compute_greeks(
    FpeCompute* ptr,
    const double* K,
    int32_t n_K,
    double rel_s,
    double rel_v,
    struct FpeGreeksResult* result
);

/** One-shot pricing: compute pipeline and price + Greeks in a single call.
 *
 *  All model parameters are the same as fpe_compute_create().
 *  @param result Filled with price, delta, gamma, vega arrays
 */
extern void fpe_price_oneshot(
    double kappa, double theta, double sigma, double rho,
    double r, double T, double S0, double V0,
    const double* K, int32_t n_K,
    double barrier, int32_t option_type,
    int32_t n_s, int32_t n_v,
    int32_t num_insert,
    double s_min, double s_max,
    struct FpeOneshotResult* result
);

/** Free a FpeVecResult allocated by the engine. */
extern void fpe_compute_free_vec(struct FpeVecResult* r);
/** Free a FpeVec2Result allocated by the engine. */
extern void fpe_compute_free_vec2(struct FpeVec2Result* r);
/** Free a FpeGridPtsResult allocated by the engine. */
extern void fpe_compute_free_grid_pts(struct FpeGridPtsResult* r);
/** Free a FpeMatResult allocated by the engine. */
extern void fpe_compute_free_mat(struct FpeMatResult* r);
/** Free a FpeGreeksResult allocated by the engine. */
extern void fpe_compute_free_greeks(struct FpeGreeksResult* r);
/** Free a FpeOneshotResult allocated by the engine. */
extern void fpe_compute_free_oneshot(struct FpeOneshotResult* r);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(struct FpeVecResult) == 16, "FpeVecResult layout mismatch");
static_assert(sizeof(struct FpeVec2Result) == 32, "FpeVec2Result layout mismatch");
static_assert(sizeof(struct FpeGridPtsResult) == 48, "FpeGridPtsResult layout mismatch");
static_assert(sizeof(struct FpeMatResult) == 16, "FpeMatResult layout mismatch");
static_assert(sizeof(struct FpeGreeksResult) == 32, "FpeGreeksResult layout mismatch");
static_assert(sizeof(struct FpeOneshotResult) == 40, "FpeOneshotResult layout mismatch");
#endif

#endif
