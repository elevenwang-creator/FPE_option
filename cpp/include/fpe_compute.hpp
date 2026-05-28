#ifndef FPE_COMPUTE_HPP
#define FPE_COMPUTE_HPP

#include "fpe_engine.h"
#include <vector>
#include <optional>
#include <cstddef>
#include <cstdio>

namespace fpe {

struct KnotsResult {
    std::vector<double> s;
    std::vector<double> v;
};

struct GridPointsResult {
    std::vector<double> s;
    std::vector<double> v;
    std::vector<double> s_weights;
    std::vector<double> v_weights;
};

struct GreeksResult {
    std::vector<double> delta;
    std::vector<double> gamma;
    std::vector<double> vega;
};

struct OneshotResult {
    std::vector<double> price;
    std::vector<double> delta;
    std::vector<double> gamma;
    std::vector<double> vega;
};

class FpeCompute {
    ::FpeCompute* ptr_;
    mutable std::optional<KnotsResult> knots_;
    mutable std::optional<GridPointsResult> grid_points_;
    mutable std::optional<std::vector<double>> initial_condition_;
    mutable std::optional<std::vector<std::vector<double>>> solve_;
    mutable std::optional<std::vector<std::vector<double>>> pdf_;

    static std::vector<double> vec_from_c(FpeVecResult&& r) {
        std::vector<double> v;
        if (r.data && r.len > 0) {
            v.assign(r.data, r.data + r.len);
        }
        fpe_compute_free_vec(&r);
        return v;
    }

    static KnotsResult knots_from_c(FpeVec2Result&& r) {
        KnotsResult k;
        if (r.s_data && r.s_len > 0) {
            k.s.assign(r.s_data, r.s_data + r.s_len);
        }
        if (r.v_data && r.v_len > 0) {
            k.v.assign(r.v_data, r.v_data + r.v_len);
        }
        fpe_compute_free_vec2(&r);
        return k;
    }

    static GridPointsResult grid_from_c(FpeGridPtsResult&& r) {
        GridPointsResult g;
        if (r.s_data && r.s_len > 0) {
            g.s.assign(r.s_data, r.s_data + r.s_len);
            g.s_weights.assign(r.sw_data, r.sw_data + r.s_len);
        }
        if (r.v_data && r.v_len > 0) {
            g.v.assign(r.v_data, r.v_data + r.v_len);
            g.v_weights.assign(r.vw_data, r.vw_data + r.v_len);
        }
        fpe_compute_free_grid_pts(&r);
        return g;
    }

    static std::vector<std::vector<double>> mat_from_c(FpeMatResult&& r) {
        std::vector<std::vector<double>> m;
        if (!r.data || r.n_rows == 0 || r.n_cols == 0) {
            fpe_compute_free_mat(&r);
            return m;
        }
        m.resize(r.n_rows);
        for (int32_t i = 0; i < r.n_rows; i++) {
            m[i].assign(r.data + i * r.n_cols, r.data + (i + 1) * r.n_cols);
        }
        fpe_compute_free_mat(&r);
        return m;
    }

    static GreeksResult greeks_from_c(FpeGreeksResult&& r) {
        GreeksResult g;
        if (r.delta && r.len > 0) {
            g.delta.assign(r.delta, r.delta + r.len);
            g.gamma.assign(r.gamma, r.gamma + r.len);
            g.vega.assign(r.vega, r.vega + r.len);
        }
        fpe_compute_free_greeks(&r);
        return g;
    }

public:
    FpeCompute(
        double kappa, double theta, double sigma, double rho, double r,
        double T, double S0, double V0,
        int32_t n_s, int32_t n_v, double barrier, int32_t option_type,
        int32_t num_insert = 50
    ) : ptr_(nullptr) {
        ptr_ = fpe_compute_create(
            kappa, theta, sigma, rho, r, T, S0, V0,
            n_s, n_v, barrier, option_type, num_insert
        );
        if (!ptr_) {
            fprintf(stderr, "FpeCompute: failed to create pipeline\n");
        }
    }

    ~FpeCompute() {
        if (ptr_) {
            fpe_compute_destroy(ptr_);
            ptr_ = nullptr;
        }
    }

    FpeCompute(const FpeCompute&) = delete;
    FpeCompute& operator=(const FpeCompute&) = delete;

    FpeCompute(FpeCompute&& o) noexcept
        : ptr_(o.ptr_) {
        o.ptr_ = nullptr;
        knots_ = std::move(o.knots_);
        grid_points_ = std::move(o.grid_points_);
        initial_condition_ = std::move(o.initial_condition_);
        solve_ = std::move(o.solve_);
        pdf_ = std::move(o.pdf_);
    }

    FpeCompute& operator=(FpeCompute&& o) noexcept {
        if (this != &o) {
            if (ptr_) fpe_compute_destroy(ptr_);
            ptr_ = o.ptr_;
            o.ptr_ = nullptr;
            knots_ = std::move(o.knots_);
            grid_points_ = std::move(o.grid_points_);
            initial_condition_ = std::move(o.initial_condition_);
            solve_ = std::move(o.solve_);
            pdf_ = std::move(o.pdf_);
        }
        return *this;
    }

    bool valid() const { return ptr_ != nullptr; }

    KnotsResult knots() const {
        if (!knots_) {
            knots_ = knots_from_c(fpe_compute_knots(ptr_));
        }
        return knots_.value();
    }

    GridPointsResult grid_points() const {
        if (!grid_points_) {
            grid_points_ = grid_from_c(fpe_compute_grid_points(ptr_));
        }
        return grid_points_.value();
    }

    std::vector<double> initial_condition() {
        if (!initial_condition_) {
            initial_condition_ = vec_from_c(fpe_compute_initial_condition(ptr_));
        }
        return initial_condition_.value();
    }

    std::vector<std::vector<double>> solve() {
        if (!solve_) {
            solve_ = mat_from_c(fpe_compute_solve(ptr_));
        }
        return solve_.value();
    }

    std::vector<std::vector<double>> pdf() {
        if (!pdf_) {
            pdf_ = mat_from_c(fpe_compute_pdf(ptr_));
        }
        return pdf_.value();
    }

    std::vector<double> price(const std::vector<double>& K) {
        return vec_from_c(fpe_compute_price(ptr_, K.data(), int32_t(K.size())));
    }

    GreeksResult greeks(const std::vector<double>& K, double rel_s = 0.01, double rel_v = 0.1) {
        return greeks_from_c(fpe_compute_greeks(ptr_, K.data(), int32_t(K.size()), rel_s, rel_v));
    }

    static OneshotResult price_oneshot(
        double kappa, double theta, double sigma, double rho, double r,
        double T, double S0, double V0,
        const std::vector<double>& K,
        double barrier, int32_t option_type,
        int32_t n_s, int32_t n_v,
        int32_t num_insert = 50
    ) {
        auto raw = fpe_price_oneshot(
            kappa, theta, sigma, rho, r, T, S0, V0,
            K.data(), int32_t(K.size()),
            barrier, option_type, n_s, n_v, num_insert
        );
        OneshotResult o;
        if (raw.price && raw.len > 0) {
            o.price.assign(raw.price, raw.price + raw.len);
            o.delta.assign(raw.delta, raw.delta + raw.len);
            o.gamma.assign(raw.gamma, raw.gamma + raw.len);
            o.vega.assign(raw.vega, raw.vega + raw.len);
        }
        fpe_compute_free_oneshot(&raw);
        return o;
    }
};

}

#endif
