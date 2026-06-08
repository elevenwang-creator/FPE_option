#include <cstdio>
#include <vector>
#include <chrono>
#include "fpe_compute.hpp"

using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    const char* label;
    double ms;
};

static std::vector<BenchResult> benches;

#define BENCH(label, expr) do { \
    auto _t0 = Clock::now(); \
    expr; \
    auto _t1 = Clock::now(); \
    double _ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count(); \
    benches.push_back({label, _ms}); \
    printf("  [%s] %.1f ms\n", label, _ms); fflush(stdout); \
} while(0)

int main() {
    printf("=== FPE Compute Pipeline - C++ API (Benchmark) ===\n\n"); fflush(stdout);

    std::vector<double> K = {65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0};

    printf("[1] Create pipeline - European Call\n"); fflush(stdout);
    fpe::FpeCompute eu(
        1.2, 0.05, 0.35, -0.4, 0.1, 0.6, 60.0, 0.1, 16, 16, 0.0, 8);
    if (!eu.valid()) {
        printf("ERROR: failed to create European pipeline\n");
        return 1;
    }

    printf("[2] Create pipeline - Down-and-Out Barrier Call\n"); fflush(stdout);
    fpe::FpeCompute bar(
        1.2, 0.05, 0.35, -0.4, 0.1, 0.6, 60.0, 0.1, 16, 16, 50.0, 2,
        251, 0.0, 150.0);
    if (!bar.valid()) {
        printf("ERROR: failed to create barrier pipeline\n");
        return 1;
    }

    // 38x38 pipeline for benchmark comparison (matching Mojo native bench config)
    printf("\n[12] Create pipeline - 38x38 Barrier Benchmark\n"); fflush(stdout);
    fpe::FpeCompute bench(
        1.2, 0.05, 0.35, -0.4, 0.1, 0.6, 60.0, 0.1, 38, 38, 50.0, 2,
        251, 0.0, 150.0);
    if (!bench.valid()) {
        printf("ERROR: failed to create bench pipeline\n");
        return 1;
    }

    std::vector<double> prices_bench;
    BENCH("bench_price", prices_bench = bench.price(K));
    printf("[38x38 Barrier Benchmark]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f\n", K[i], prices_bench[i]);

    fpe::GreeksResult g_bench;
    BENCH("bench_greeks", g_bench = bench.greeks(K));
    printf("[38x38 Barrier Greeks]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: delta=%.6f gamma=%.6f vega=%.6f\n",
            K[i], g_bench.delta[i], g_bench.gamma[i], g_bench.vega[i]);
    fflush(stdout);

    printf("\n[3] Knots\n"); fflush(stdout);
    fpe::KnotsResult k_eu;
    BENCH("eu_knots", k_eu = eu.knots());
    printf("  [European] s_knots(%zu):", k_eu.s.size());
    for (size_t i = 0; i < k_eu.s.size() && i < 6; i++) printf(" %.4f", k_eu.s[i]);
    if (k_eu.s.size() > 6) printf(" ...");
    printf("\n");

    fpe::KnotsResult k_bar;
    BENCH("bar_knots", k_bar = bar.knots());
    printf("  [Barrier] s_knots(%zu):", k_bar.s.size());
    for (size_t i = 0; i < k_bar.s.size() && i < 6; i++) printf(" %.4f", k_bar.s[i]);
    if (k_bar.s.size() > 6) printf(" ...");
    printf("\n"); fflush(stdout);

    printf("\n[4] Grid Points\n"); fflush(stdout);
    fpe::GridPointsResult gp;
    BENCH("eu_grid_pts", gp = eu.grid_points());
    printf("  [European] %zu s-pts, %zu v-pts\n", gp.s.size(), gp.v.size());

    printf("\n[5] Initial Condition\n"); fflush(stdout);
    std::vector<double> q0;
    BENCH("eu_init_cond", q0 = eu.initial_condition());
    printf("  [European] q0 length=%zu, q0[0]=%.6f, q0[%zu]=%.6f\n",
        q0.size(), q0[0], q0.size() - 1, q0[q0.size() - 1]);
    fflush(stdout);

    printf("\n[6] Solve (this may take a moment)...\n"); fflush(stdout);
    std::vector<std::vector<double>> sol;
    BENCH("eu_solve", sol = eu.solve());
    printf("  [European] solution: %zu time steps, %zu DOF per step\n",
        sol.size(), sol.empty() ? 0 : sol[0].size());
    fflush(stdout);

    printf("\n[7] PDF\n"); fflush(stdout);
    std::vector<std::vector<double>> pdf;
    BENCH("eu_pdf", pdf = eu.pdf());
    printf("  [European] PDF: %zu rows x %zu cols\n",
        pdf.size(), pdf.empty() ? 0 : pdf[0].size());

    printf("\n[8] Pricing\n"); fflush(stdout);
    std::vector<double> prices_eu;
    BENCH("eu_price", prices_eu = eu.price(K));
    printf("[European Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f\n", K[i], prices_eu[i]);

    std::vector<double> prices_bar;
    BENCH("bar_price", prices_bar = bar.price(K));
    printf("[Down-and-Out Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f\n", K[i], prices_bar[i]);
    fflush(stdout);

    printf("\n[9] Greeks\n"); fflush(stdout);
    fpe::GreeksResult g_eu;
    BENCH("eu_greeks", g_eu = eu.greeks(K));
    printf("[European Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: delta=%.6f gamma=%.6f vega=%.6f\n",
            K[i], g_eu.delta[i], g_eu.gamma[i], g_eu.vega[i]);

    fpe::GreeksResult g_bar;
    BENCH("bar_greeks", g_bar = bar.greeks(K));
    printf("[Down-and-Out Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: delta=%.6f gamma=%.6f vega=%.6f\n",
            K[i], g_bar.delta[i], g_bar.gamma[i], g_bar.vega[i]);
    fflush(stdout);

    printf("\n[10] One-shot pricing (PricingEngine)\n"); fflush(stdout);
    fpe::OneshotResult os;
    BENCH("oneshot", os = fpe::FpeCompute::price_oneshot(
        1.2, 0.05, 0.35, -0.4, 0.1, 0.6, 60.0, 0.1, K, 0.0, 8, 16, 16));
    printf("[One-shot European Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f delta=%.6f gamma=%.6f vega=%.6f\n",
            K[i], os.price[i], os.delta[i], os.gamma[i], os.vega[i]);

    printf("\n[11] Cross-verification: C++ vs Python reference\n"); fflush(stdout);
    struct Ref { double K; double py_price; };
    Ref refs[] = {
        {65.0, 4.749455}, {70.0, 2.941798}, {75.0, 1.719919},
        {80.0, 0.962836}, {85.0, 0.535476}, {90.0, 0.321226},
        {95.0, 0.230060}, {100.0, 0.200840},
    };
    bool all_ok = true;
    for (auto& r : refs) {
        double cpp_p = prices_eu[0];
        for (size_t i = 0; i < K.size(); i++) {
            if (K[i] == r.K) { cpp_p = prices_eu[i]; break; }
        }
        double rel = (cpp_p - r.py_price) / r.py_price;
        bool ok = (rel > -0.01 && rel < 0.01);
        printf("  K=%.1f: C++=%.6f Py=%.6f rel_diff=%.4f%% %s\n",
            r.K, cpp_p, r.py_price, rel * 100.0, ok ? "OK" : "MISMATCH");
        if (!ok) all_ok = false;
    }

    printf("\n=== Benchmark Summary ===\n");
    double total = 0;
    for (auto& b : benches) {
        printf("  %-20s %10.1f ms\n", b.label, b.ms);
        total += b.ms;
    }
    printf("  %-20s %10.1f ms\n", "TOTAL", total);
    printf("\nVerification: %s\n", all_ok ? "ALL PASS" : "MISMATCH DETECTED");
    printf("\nDone.\n");
    return all_ok ? 0 : 1;
}
