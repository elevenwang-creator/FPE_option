#include <cstdio>
#include <vector>
#include "fpe_compute.hpp"

int main() {
    printf("=== FPE Compute Pipeline - C++ API ===\n\n");

    std::vector<double> K = {65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0};

    printf("[1] Create pipeline - European Call\n");
    fpe::FpeCompute eu(
        1.2, 0.05, 0.35, -0.4, 0.1, 0.6, 60.0, 0.1, 16, 16, 0.0, 8);
    if (!eu.valid()) {
        printf("ERROR: failed to create European pipeline\n");
        return 1;
    }

    printf("[2] Create pipeline - Down-and-Out Barrier Call\n");
    fpe::FpeCompute bar(
        1.2, 0.05, 0.35, -0.4, 0.1, 0.6, 60.0, 0.1, 16, 16, 50.0, 2);
    if (!bar.valid()) {
        printf("ERROR: failed to create barrier pipeline\n");
        return 1;
    }

    printf("\n[3] Knots\n");
    auto k_eu = eu.knots();
    printf("[European] s_knots(%zu):", k_eu.s.size());
    for (size_t i = 0; i < k_eu.s.size() && i < 6; i++) printf(" %.4f", k_eu.s[i]);
    if (k_eu.s.size() > 6) printf(" ...");
    printf("\n");

    auto k_bar = bar.knots();
    printf("[Barrier] s_knots(%zu):", k_bar.s.size());
    for (size_t i = 0; i < k_bar.s.size() && i < 6; i++) printf(" %.4f", k_bar.s[i]);
    if (k_bar.s.size() > 6) printf(" ...");
    printf("\n");

    printf("\n[4] Grid Points\n");
    auto gp = eu.grid_points();
    printf("[European] %zu s-pts, %zu v-pts\n", gp.s.size(), gp.v.size());

    printf("\n[5] Initial Condition\n");
    auto q0 = eu.initial_condition();
    printf("[European] q0 length=%zu, q0[0]=%.6f, q0[%zu]=%.6f\n",
           q0.size(), q0[0], q0.size() - 1, q0[q0.size() - 1]);

    printf("\n[6] Solve\n");
    auto sol = eu.solve();
    printf("[European] solution: %zu time steps, %zu DOF per step\n",
           sol.size(), sol.empty() ? 0 : sol[0].size());

    printf("\n[7] PDF\n");
    auto pdf = eu.pdf();
    printf("[European] PDF: %zu rows x %zu cols\n",
           pdf.size(), pdf.empty() ? 0 : pdf[0].size());

    printf("\n[8] Pricing\n");
    auto prices_eu = eu.price(K);
    printf("[European Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f\n", K[i], prices_eu[i]);

    auto prices_bar = bar.price(K);
    printf("[Down-and-Out Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f\n", K[i], prices_bar[i]);

    printf("\n[9] Greeks\n");
    auto g_eu = eu.greeks(K);
    printf("[European Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: delta=%.6f gamma=%.6f vega=%.6f\n",
               K[i], g_eu.delta[i], g_eu.gamma[i], g_eu.vega[i]);

    auto g_bar = bar.greeks(K);
    printf("[Down-and-Out Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: delta=%.6f gamma=%.6f vega=%.6f\n",
               K[i], g_bar.delta[i], g_bar.gamma[i], g_bar.vega[i]);

    printf("\n[10] One-shot pricing (PricingEngine)\n");
    auto os = fpe::FpeCompute::price_oneshot(
        1.2, 0.05, 0.35, -0.4, 0.1, 0.6, 60.0, 0.1, K, 0.0, 8, 16, 16);
    printf("[One-shot European Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f delta=%.6f gamma=%.6f vega=%.6f\n",
               K[i], os.price[i], os.delta[i], os.gamma[i], os.vega[i]);

    printf("\nDone.\n");
    return 0;
}
