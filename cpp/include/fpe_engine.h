#ifndef FPE_ENGINE_H
#define FPE_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FpeCompute FpeCompute;

struct FpeVecResult {
    double* data;
    int32_t len;
};

struct FpeVec2Result {
    double* s_data;
    int32_t s_len;
    double* v_data;
    int32_t v_len;
};

struct FpeGridPtsResult {
    double* s_data;
    int32_t s_len;
    double* v_data;
    int32_t v_len;
    double* sw_data;
    double* vw_data;
};

struct FpeMatResult {
    double* data;
    int32_t n_rows;
    int32_t n_cols;
};

struct FpeGreeksResult {
    double* delta;
    double* gamma;
    double* vega;
    int32_t len;
};

struct FpeOneshotResult {
    double* price;
    double* delta;
    double* gamma;
    double* vega;
    int32_t len;
};

extern FpeCompute* fpe_compute_create(
    double kappa, double theta, double sigma, double rho,
    double r, double T, double S0, double V0,
    int32_t n_s, int32_t n_v,
    double barrier, int32_t option_type,
    int32_t num_insert
);

extern void fpe_compute_destroy(FpeCompute* ptr);

extern struct FpeVec2Result fpe_compute_knots(FpeCompute* ptr);

extern struct FpeGridPtsResult fpe_compute_grid_points(FpeCompute* ptr);

extern struct FpeVecResult fpe_compute_initial_condition(FpeCompute* ptr);

extern struct FpeMatResult fpe_compute_solve(FpeCompute* ptr);

extern struct FpeMatResult fpe_compute_pdf(FpeCompute* ptr);

extern struct FpeVecResult fpe_compute_price(
    FpeCompute* ptr,
    const double* K,
    int32_t n_K
);

extern struct FpeGreeksResult fpe_compute_greeks(
    FpeCompute* ptr,
    const double* K,
    int32_t n_K,
    double rel_s,
    double rel_v
);

extern struct FpeOneshotResult fpe_price_oneshot(
    double kappa, double theta, double sigma, double rho,
    double r, double T, double S0, double V0,
    const double* K, int32_t n_K,
    double barrier, int32_t option_type,
    int32_t n_s, int32_t n_v,
    int32_t num_insert
);

extern void fpe_compute_free_vec(struct FpeVecResult* r);
extern void fpe_compute_free_vec2(struct FpeVec2Result* r);
extern void fpe_compute_free_grid_pts(struct FpeGridPtsResult* r);
extern void fpe_compute_free_mat(struct FpeMatResult* r);
extern void fpe_compute_free_greeks(struct FpeGreeksResult* r);
extern void fpe_compute_free_oneshot(struct FpeOneshotResult* r);

#ifdef __cplusplus
}
#endif

#endif
