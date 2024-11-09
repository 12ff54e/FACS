#include <algorithm>  // lower_bound, upper_bound
#include <cmath>      // ceil, floor
#include <filesystem>
#include <iostream>
#include <numbers>  // pi
#include <unordered_map>

#include "Floquet.h"
#include "equilibrium.h"
#include "integrator.h"

#define ZQ_TIMER_IMPLEMENTATION
#include "timer.h"

// inject hash function for pair of int
namespace std {
template <>
struct hash<pair<int, int>> {
    auto operator()(const pair<int, int>& p) const noexcept {
        return hash<int>{}(p.first) ^ (hash<int>{}(p.second) << 1);
    }
};
};  // namespace std

int main(int argc, char** argv) {
    using namespace std::complex_literals;

    // constexpr double a = 0.1;
    // const auto q = [](double r) { return 1.71 + 0.16 * r * r / (a * a); };
    // const auto dp = [](double r) { return 0.; };  // ad hoc expression

    // const AnalyticEquilibrium equilibrium(q, dp);

    if (argc < 2) { return EPERM; }
    std::string gfile_path = argv[1];

    auto& timer = Timer::get_timer();
    timer.start("Read gfile");

    std::ifstream gfile(gfile_path);
    if (!gfile.is_open()) {
        std::cerr << "Can not open g-file.\n";
        return ENOENT;
    }
    GFileRawData gfile_data;
    gfile >> gfile_data;
    if (!gfile_data.is_complete()) {
        std::cerr << "Can not parse g-file.\n";
        return 0;
    }
    gfile.close();

    const std::size_t radial_count = 384;
    const std::size_t omega_count = 250;
    // this value is normalized to $v_{A,0}/(q_min*R_0)$, making $\omega$
    // to reach least somewhere between 2nd and 3rd gap for all radial
    // position.
    constexpr double max_omega = 1.2;

    const NumericEquilibrium<double> equilibrium(gfile_data, radial_count, 300);
    const auto r_max = equilibrium.psi_at_wall();
    const auto delta_r = r_max / radial_count;

    auto get_psi = [&](std::size_t idx) {
        static const double delta =
            equilibrium.psi_at_wall() / (radial_count * radial_count);
        return idx * idx * delta;
    };

    double q_min = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < radial_count; ++i) {
        q_min = std::min(q_min, equilibrium.safety_factor(get_psi(i + 1)));
    }

    // toroidal mode numbers
    std::vector<int> ns{8};
    // poloidal mode numbers
    std::vector<std::vector<std::pair<int, int>>> m_ranges(radial_count);
    std::vector<std::vector<double>> continuum(radial_count);

    for (std::size_t i = 0; i < radial_count; ++i) {
        timer.pause_last_and_start_next("Calculate Floquet exponent");

        // TODO: Need some refinement around stability boundary
        const auto r = get_psi(i + 1);
        const auto local_q = equilibrium.safety_factor(r);

        std::vector<std::complex<double>> local_nu;
        local_nu.reserve(omega_count);

        const auto domega =
            local_q / q_min * max_omega / static_cast<double>(omega_count - 1);
        for (std::size_t j = 0; j < omega_count; ++j) {
            // convert between global and local normalization of $\omega$
            auto omega2 = std::pow(domega * static_cast<double>(j), 2);
            auto potential = [omega2, r, &eq = equilibrium](double theta) {
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

        timer.pause_last_and_start_next("Solve for omega");

        // calculate omega for each pair of mode numbers (n, m)
        // change normalization of domega to v_{A,0}/R_0 here
        const auto domega_global = domega / local_q;
        for (std::size_t n_idx = 0; n_idx < ns.size(); ++n_idx) {
            auto n = ns[n_idx];
            std::size_t m_num = 0;
            const int m_lower = std::ceil(n * local_q - local_nu.back().real());
            const int m_upper =
                std::floor(n * local_q + local_nu.back().real());
            m_ranges[i].emplace_back(m_lower, m_upper);

            for (int m = m_lower; m <= m_upper; ++m) {
                const double kp = n * local_q - m;
                auto it = std::lower_bound(
                    local_nu.begin(), local_nu.end(), std::abs(kp),
                    [](auto nu_c, auto k) { return std::real(nu_c) < k; });
                if (kp > 0. && 2 * kp - std::round(2 * kp) < 1.e-7) {
                    // accumulation point, nq-m>0 branch
                    it = std::upper_bound(
                        local_nu.begin(), local_nu.end(), std::abs(kp),
                        [](auto k, auto nu_c) { return k < std::real(nu_c); });
                }
                if (it == local_nu.begin()) {
                    continuum[i].push_back(0.);
                } else if (it != local_nu.end()) {
                    continuum[i].push_back(
                        (it - local_nu.begin() +
                         (std::abs(kp) - (it - 1)->real()) /
                             (it->real() - (it - 1)->real())) *
                        domega_global);
                }
            }
        }
    }

    timer.pause_last_and_start_next("Sort points into lines");

    // sort by (n,m) or (n, \nu) to individual lines
    std::unordered_map<std::pair<int, int>, std::vector<std::array<double, 2>>>
        lines;
    const bool sort_by_m = false;
    if (sort_by_m) {
        // To be deprecated
        for (std::size_t i = 0; i < radial_count; ++i) {
            std::size_t offset = 0;
            for (std::size_t j = 0; j < ns.size(); ++j) {
                auto [m_lower, m_upper] = m_ranges[i][j];
                for (int k = 0; k <= m_upper - m_lower; ++k) {
                    lines[std::make_pair(ns[j], k + m_lower)].push_back(
                        {r_max * static_cast<double>(i) /
                             static_cast<double>(radial_count),
                         continuum[i][k + offset]});
                }
                offset += m_upper - m_lower + 1;
            }
        }
    } else {
        // sort by Floquet exponent
        for (std::size_t i = 0; i < radial_count; ++i) {
            // Pay attention to radial coordinate, it should be the same as that
            // used above
            // TODO: Ensure they are the same
            const auto psi = get_psi(i + 1);
            std::size_t offset = 0;
            for (std::size_t j = 0; j < ns.size(); ++j) {
                auto [m_lower, m_upper] = m_ranges[i][j];
                for (int k = 0; k <= m_upper - m_lower; ++k) {
                    const int kp = std::floor(std::abs(
                        .5 +
                        std::floor(2 * (ns[j] * equilibrium.safety_factor(psi) -
                                        (k + m_lower)))));
                    lines[std::make_pair(ns[j], kp)].push_back(
                        {equilibrium.minor_radius(psi),
                         continuum[i][k + offset]});
                }
                offset += m_upper - m_lower + 1;
            }
        }
    }

    timer.pause_last_and_start_next("Output");

    // output
    auto output_file_name =
        std::string{"continuum-"} +
        std::filesystem::path(gfile_path).filename().string();
    std::ofstream output(output_file_name);
    if (!output.is_open()) {
        std::cerr << "Failed to open " << std::quoted(output_file_name)
                  << " for write.";
        return ENOENT;
    }
    for (auto& line : lines) {
        const auto& [nm, coords] = line;
        output << nm.first << ' ' << nm.second << ' ';
        for (auto pt : coords) { output << pt[0] << ' ' << pt[1] << ' '; }
        output << '\n';
    }

    output.close();

    timer.pause();
    timer.print();

    return 0;
}
