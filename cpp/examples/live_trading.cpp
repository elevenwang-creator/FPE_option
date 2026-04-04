#include "../include/fpe_engine.h"

#include <stdio.h>

int main() {
    if (fpe_init() != 0) {
        fprintf(stderr, "Failed to initialize FPE engine\n");
        return 1;
    }

    double price = 0.0;
    double delta = 0.0;
    double gamma = 0.0;
    int result = fpe_price_single(100.0, 105.0, 0.1, 120.0, 0, 12345ULL, &price, &delta, &gamma);

    if (result == 0) {
        printf("Price: %.6f, Delta: %.6f, Gamma: %.6f\n", price, delta, gamma);
    } else {
        fprintf(stderr, "Pricing failed (cache miss — run fpe_solve_fpe first)\n");
    }

    fpe_destroy();
    return 0;
}
