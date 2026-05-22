#ifndef FPE_ENGINE_H
#define FPE_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double price;
    double delta;
    double gamma;
    double vega;
    int32_t success;
} FpePriceResult;

#define FPE_ERR_OK              0
#define FPE_ERR_NULL_K         -1
#define FPE_ERR_NULL_OUT       -2
#define FPE_ERR_INVALID_NSTRIKES -3
#define FPE_ERR_BUFFER_TOO_SMALL -4
#define FPE_ERR_INVALID_PARAMS -5
#define FPE_ERR_SOLVER_FAILED  -6
#define FPE_ERR_TOO_MANY_STRIKES -7

#define FPE_MAX_STRIKES 1024

extern int32_t fpe_price(
    double kappa,
    double theta,
    double sigma,
    double rho,
    double r,
    double T,
    double S0,
    double V0,
    const double* K,
    int32_t n_strikes,
    double barrier,
    int32_t option_type,
    int32_t n_s,
    int32_t n_v,
    double rtol,
    double atol,
    FpePriceResult* out,
    int32_t out_capacity
);

#ifdef __cplusplus
}
#endif

#endif
