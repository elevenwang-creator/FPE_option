#ifndef FPE_ENGINE_H
#define FPE_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    double price;
    double delta;
    double gamma;
    double vega;
    bool success;
} FpePriceResult;

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
    FpePriceResult* out
);

#endif
