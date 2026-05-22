#include <cstdio>
#include <cmath>
#include "fpe_engine.h"

static void print_results(const char* label, int32_t n, const double K[], FpePriceResult results[]) {
    if (n < 0) {
        printf("[%s] ERROR: code %d\n", label, n);
        return;
    }
    printf("[%s]:\n", label);
    for (int32_t i = 0; i < n; i++) {
        printf("  K=%.1f: price=%.6f delta=%.6f gamma=%.6f vega=%.6f success=%d\n",
            K[i], results[i].price, results[i].delta,
            results[i].gamma, results[i].vega, results[i].success);
    }
}

int main() {
    printf("=== FPE Option Pricing - C API ===\n\n");

    double K[] = {65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0};
    int32_t n_strikes = 8;
    FpePriceResult results[8];
    int32_t n;

    printf("[1] European Call (barrier=0, option_type=8):\n");
    n = fpe_price(
        1.2, 0.05, 0.35, -0.4,
        0.1, 0.6, 60.0, 0.1,
        K, n_strikes,
        0.0, 8,
        38, 38,
        1e-4, 1e-6,
        results, n_strikes
    );
    print_results("European Call", n, K, results);

    printf("\n[2] Up-and-Out Call (barrier=80, option_type=6):\n");
    n = fpe_price(
        1.2, 0.05, 0.35, -0.4,
        0.1, 0.6, 60.0, 0.1,
        K, n_strikes,
        80.0, 6,
        38, 38,
        1e-4, 1e-6,
        results, n_strikes
    );
    print_results("Up-and-Out Call", n, K, results);

    printf("\n[3] Down-and-Out Call (barrier=50, option_type=2):\n");
    n = fpe_price(
        1.2, 0.05, 0.35, -0.4,
        0.1, 0.6, 60.0, 0.1,
        K, n_strikes,
        50.0, 2,
        38, 38,
        1e-4, 1e-6,
        results, n_strikes
    );
    print_results("Down-and-Out Call", n, K, results);

    printf("\n[4] Error handling - NULL K pointer:\n");
    n = fpe_price(
        1.2, 0.05, 0.35, -0.4,
        0.1, 0.6, 60.0, 0.1,
        NULL, n_strikes,
        0.0, 8,
        38, 38,
        1e-4, 1e-6,
        results, n_strikes
    );
    print_results("NULL K test", n, K, results);

    printf("\n[5] Error handling - buffer too small:\n");
    n = fpe_price(
        1.2, 0.05, 0.35, -0.4,
        0.1, 0.6, 60.0, 0.1,
        K, n_strikes,
        0.0, 8,
        38, 38,
        1e-4, 1e-6,
        results, 1
    );
    print_results("Buffer too small test", n, K, results);

    printf("\nDone.\n");
    return 0;
}
