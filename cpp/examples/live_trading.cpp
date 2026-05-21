#include <cstdio>
#include <cmath>
#include "fpe_engine.h"

int main() {
    double K[] = {95.0, 100.0, 105.0};
    int32_t n_strikes = 3;
    FpePriceResult results[3];

    int32_t n = fpe_price(
        1.2, 0.05, 0.35, -0.4,
        0.05, 0.5, 100.0, 0.1,
        K, n_strikes,
        120.0, 6,
        8, 8,
        1e-4, 1e-6,
        results
    );

    for (int32_t i = 0; i < n; i++) {
        printf("K=%.1f: price=%.4f delta=%.4f gamma=%.4f vega=%.4f success=%d\n",
            K[i], results[i].price, results[i].delta,
            results[i].gamma, results[i].vega, results[i].success);
    }

    return 0;
}
