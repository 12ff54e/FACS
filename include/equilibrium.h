#ifndef EQUILIBRIUM_H
#define EQUILIBRIUM_H

#include "gFileRawData.h"
#include "spdata.h"

// For axisymmetric equilibrium, we need g^{rr} and J for the continuum
// equation. For usability I decide to provide an analytic eq using s-alpha
// model, and a numeric eq constructed from EFIT g-file.

template <typename FS, typename FD, typename T = double>
struct AnalyticEquilibrium {
    using value_type = T;

    AnalyticEquilibrium(const FS& safety_factor, const FD& ddelta_dr)
        : q(safety_factor), dp(ddelta_dr) {}

    // - \frac{\partial^2\sqrt{g^{rr}}}{\partial\theta^2}/\sqrt{g^{rr}}
    // \approx \Delta^\prime\cos\theta + \mathcal{O}\left(\epsilon^2\right)
    auto radial_func(value_type r, value_type theta) const {
        return dp(r) * std::cos(theta);
    }

    // J^2/q^2 \approx
    // R_0^2\left(1+\4*\frac{r}{R_0}*\cos\left(\theta\right)\right) +
    // \mathcal{O}\left(\epsilon^2\right),
    // q^2 is part of local normalization of frequency
    auto j_func(value_type r, value_type theta) const {
        return 1. + 4. * r * std::cos(theta);
    }

    auto safety_factor(value_type r) const { return q(r); }

    const FS& q;
    const FD& dp;
};

template <typename T>
struct NumericEquilibrium : private Spdata<T> {
    using value_type = T;

    using Spdata<value_type>::intp_data;

    NumericEquilibrium(const GFileRawData& g_file_data,
                       std::size_t radial_grid_num,
                       std::size_t poloidal_grid_num,
                       value_type psi_ratio = .98)
        : Spdata<value_type>(g_file_data,
                             radial_grid_num,
                             poloidal_grid_num,
                             false,
                             radial_grid_num < 256 ? radial_grid_num : 256,
                             psi_ratio) {}
    // No need to construct too many magnetic surfaces, 256 should be more than
    // necessary

    auto radial_func(value_type r, value_type theta) const {
        return -intp_data().intp_2d[4](r, theta);
    }

    auto j_func(value_type r, value_type theta) const {
        const auto j = intp_data().intp_2d[3](r, theta);
        const auto q = intp_data().intp_1d[0](r);
        return std::pow(j / q, 2);
    }

    auto safety_factor(value_type r) const { return intp_data().intp_1d[0](r); }

    auto psi_at_wall() const noexcept {
        return intp_data().psi_sample_for_output.back();
    }

    // defined as \sqrt{\psi_t/\psi_{t,\mathrm{wall}}}
    auto minor_radius(value_type psi) const {
        const auto& psi_t = intp_data().intp_1d[5];
        const auto psi_min = intp_data().psi_sample_for_output.front();
        if (psi < psi_min) {
            return std::sqrt(psi / psi_min * psi_t(psi_min) /
                             psi_t(psi_at_wall()));
        }
        return std::sqrt(psi_t(psi) / psi_t(psi_at_wall()));
    }

   private:
    Spdata<value_type> spdata_;
};

#endif  // EQUILIBRIUM_H
