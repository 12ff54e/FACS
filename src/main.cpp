#include <cmath>  // ceil, floor
#include <iostream>
#include <numbers>

#include "Floquet.h"
#include "equilibrium.h"
#include "integrator.h"
#include "intp.h"

int main() {
    using namespace std::complex_literals;

    constexpr double a = 0.2;
    const auto q = [](double r) { return 1.71 + 0.16 * r * r / (a * a); };
    const auto dp = [](double r) { return 0.; };  // ad hoc expression

    const auto q_min = q(0.);

    const AnalyticEquilibrium ITPA_EQ(q, dp);

    constexpr std::size_t radial_count = 100;
    constexpr std::size_t omega_count = 300;
    // this value is normalized to $v_{A,0}/(q_min*R_0)$, somewhere between 2nd
    // and 3rd gap for all r, in ITPA case
    constexpr double max_omega = 1.2;

    // toroidal mode numbers
    std::array ns{2, 3, 5, 6, 8, 11, 13};
    std::vector<std::vector<double>> continuum(radial_count);
    std::vector<std::array<std::size_t, ns.size()>> m_count(radial_count);

    for (std::size_t i = 0; i < radial_count; ++i) {
        // TODO: Need some refinement around stability boundary
        const auto r =
            a * static_cast<double>(i) / static_cast<double>(radial_count);
        const auto local_q = ITPA_EQ.q(r);

        std::vector<std::complex<double>> local_nu;
        local_nu.reserve(omega_count);

        const auto domega =
            local_q / q_min * max_omega / static_cast<double>(omega_count - 1);
        for (std::size_t j = 0; j < omega_count; ++j) {
            // convert between global and local normalization of $\omega$
            auto omega2 = std::pow(domega * static_cast<double>(j), 2);
            auto potential = [omega2, r, &eq = ITPA_EQ](double theta) {
                return eq.radial_func(r, theta) + omega2 * eq.j_func(r, theta);
            };
            const FloquetMatrix flo_mat(
                potential, std::numbers::pi * 2,
                static_cast<std::size_t>(100 * std::sqrt(omega2) * 2) + 100);
            // FloquetMatrix::eigenvalue always returns eigenvalue with
            // imaginary part not less than 0
            local_nu.push_back(std::log(flo_mat.eigenvalue()) /
                               (2.i * std::numbers::pi));
        }

        // adjust $\Re\nu$ according to stability region
        std::size_t order = 0;
        bool increasing = true;
        double last_real = local_nu[0].real();
        for (std::size_t j = 1; j < omega_count; ++j) {
            // normally $\Re\nu$ growth monotonic with $\omega$, but it goes
            // unchanged inside coutinuum gap, a small margin is added to
            // avoid misclassifying gap region as another stability
            // region
            if (increasing && local_nu[j].real() - last_real + 1.e-6 < 0.) {
                // entering a region where wrong branch of $\nu$ is picked
                increasing = false;
                ++order;
            } else if (!increasing &&
                       local_nu[j].real() - last_real - 1.e-6 > 0.) {
                // entering a region where wrong branch of $\nu$ is picked
                increasing = true;
                ++order;
            }

            last_real = local_nu[j].real();
            local_nu[j].real(.5 * static_cast<double>(order) +
                             (order % 2 == 0 ? last_real : .5 - last_real));
        }

        // calculate omega for each pair of mode numbers (n, m)
        // change normalization of domega to v_{A,0}/R_0 here
        const auto domega_global = domega / local_q;
        for (std::size_t n_idx = 0; n_idx < ns.size(); ++n_idx) {
            auto n = ns[n_idx];
            std::size_t m_num = 0;
            for (int m = std::ceil(n * local_q - local_nu.back().real());
                 m <= std::floor(n * local_q + local_nu.back().real()); ++m) {
                double kp = std::abs(n * local_q - m);
                auto it = std::lower_bound(
                    local_nu.begin(), local_nu.end(), kp,
                    [](auto nu_c, auto k) { return std::real(nu_c) < k; });
                if (it != local_nu.end() && it != local_nu.begin()) {
                    continuum[i].push_back(
                        (it - local_nu.begin() +
                         (kp - (it - 1)->real()) /
                             (it->real() - (it - 1)->real())) *
                        domega_global);
                    ++m_num;
                }
            }
            m_count[i][n_idx] = m_num;
        }
    }

    // output, in handcrafted JSON format
    {
        auto& os = std::cout;
        os << '[';
        for (std::size_t i = 0; i < radial_count; ++i) {
            os << '[';
            std::size_t idx = 0;
            for (std::size_t j = 0; j < ns.size(); ++j) {
                os << '[';
                for (std::size_t k = idx; k < idx + m_count[i][j]; ++k) {
                    os << continuum[i][k];
                    if (k != idx + m_count[i][j] - 1) { os << ','; }
                }
                idx += m_count[i][j];
                os << ']';
                if (j != ns.size() - 1) { os << ','; }
            }
            os << ']';
            if (i != radial_count - 1) { os << ','; }
        }
        os << ']';
    }
    return 0;
}
