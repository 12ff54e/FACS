#include <algorithm>  // lower_bound, upper_bound
#include <cmath>      // ceil, floor
#include <filesystem>
#include <iostream>
#include <map>
#include <numbers>  // pi
#include <stack>
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

// Zero criteria for float point numbers
constexpr double EPSILON = 1.e-6;

// TODO: Add command line input logic
// TODO: Adaptively choose radial location based on nq'
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

    const std::size_t radial_count = 1000;
    const std::size_t poloidal_sample_point = 300;
    const std::size_t omega_count = 250;
    const double psi_ratio = .96;
    // this value is normalized to $v_{A,0}/(q_min*R_0)$
    constexpr double max_omega = 1.2;

    const NumericEquilibrium<double> equilibrium(
        gfile_data, radial_count, poloidal_sample_point, psi_ratio);
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
    std::vector<int> ns{5};
    // poloidal mode numbers
    std::vector<std::vector<std::pair<int, int>>> m_ranges(radial_count);
    std::vector<std::vector<double>> continuum(radial_count);

    for (std::size_t i = 0; i < radial_count; ++i) {
        timer.pause_last_and_start_next("Calculate Floquet exponent");

        const auto r = get_psi(i + 1);
        const auto local_q = equilibrium.safety_factor(r);

        // convert between global and local normalization of $\omega$
        const auto max_local_omega = local_q / q_min * max_omega;
        auto calc_floquet_exp = [r, &equilibrium](auto omega) {
            const auto omega2 = omega * omega;
            const auto potential = [omega2, r,
                                    &eq = equilibrium](double theta) {
                return eq.radial_func(r, theta) + omega2 * eq.j_func(r, theta);
            };
            const FloquetMatrix flo_mat(
                potential, std::numbers::pi * 2,
                static_cast<std::size_t>(100 * std::sqrt(omega2) * 2) + 100);
            // FloquetMatrix::eigenvalue always returns eigenvalue with
            // imaginary part not less than 0
            return std::log(flo_mat.eigenvalue()) / (2.i * std::numbers::pi);
        };
        std::map<double, std::complex<double>> omega_nu_map;
        std::stack<std::pair<decltype(omega_nu_map)::iterator,
                             decltype(omega_nu_map)::iterator>>
            region_stack;

        // critiria for stoping subdivision
        constexpr double subdivision_err = 1.e-3;
        constexpr std::size_t initial_subdivision = 2;

        for (int region = 0; region < max_local_omega * initial_subdivision;
             ++region) {
            const auto omega_min = 1. / initial_subdivision * region;
            const auto omega_max =
                1. / initial_subdivision * (region + 1) > max_local_omega
                    ? max_local_omega
                    : 1. / initial_subdivision * (region + 1);
            region_stack.push(
                {omega_nu_map.emplace(omega_min, calc_floquet_exp(omega_min))
                     .first,
                 omega_nu_map.emplace(omega_max, calc_floquet_exp(omega_max))
                     .first});

            while (!region_stack.empty()) {
                const auto [pt0, pt1] = region_stack.top();
                region_stack.pop();
                const auto omega_mid = .5 * (pt0->first + pt1->first);
                const auto nu_0 = pt0->second.real();
                const auto nu_1 = pt1->second.real();
                const auto nu_actual = calc_floquet_exp(omega_mid);
                const auto it =
                    omega_nu_map.emplace_hint(pt1, omega_mid, nu_actual);
                constexpr double min_domega = 1.e-3;
                // NOTE: I don not about imaginary part of \nu, so points in gap
                // zone will be sparse. Extra subdivisions are done at
                // gap-continuum boundary.
                if ((std::abs(.5 * (nu_0 + nu_1) - nu_actual.real()) >
                         subdivision_err ||
                     (nu_0 < EPSILON != nu_1 < EPSILON) ||
                     (.5 - nu_0 < EPSILON != .5 - nu_1 < EPSILON)) &&
                    pt1->first - omega_mid > min_domega) {
                    region_stack.push({it, pt1});
                    region_stack.push({pt0, it});
                }
            }
        }

        timer.pause_last_and_start_next("Solve for omega");

        std::vector<decltype(omega_nu_map)::value_type> local_omega_nu(
            omega_nu_map.begin(), omega_nu_map.end());
        std::cout << i << ", " << local_omega_nu.size() << '\n';

        // adjust $\Re\nu$ according to stability region
        std::size_t order = 0;
        bool increasing = true;
        double last_real = local_omega_nu[0].second.real();
        for (auto& [_, nu] : local_omega_nu) {
            // normally $\Re\nu$ growth monotonic with $\omega$, but it goes
            // unchanged inside coutinuum gap, a small margin is added to
            // avoid misclassifying gap region as another stability
            // region
            // TODO: Fix these two ad-hoc check: last_real > .4, < .1
            if (increasing && nu.real() - last_real + EPSILON < 0. &&
                last_real > .4) {
                // e^{i\nu T} entering lower half plane
                increasing = false;
                ++order;
            } else if (!increasing && nu.real() - last_real - EPSILON > 0. &&
                       last_real < .1) {
                // e^{i\nu T} entering upper half plane
                increasing = true;
                ++order;
            }

            last_real = nu.real();
            nu.real(.5 * static_cast<double>(order) +
                    (order % 2 == 0 ? last_real : .5 - last_real));
        }

        // calculate omega for each pair of mode numbers (n, m)
        // change normalization of omega to v_{A,0}/R_0 here
        const auto local_max_nu = local_omega_nu.back().second.real();
        for (std::size_t n_idx = 0; n_idx < ns.size(); ++n_idx) {
            auto n = ns[n_idx];
            std::size_t m_num = 0;
            const int m_lower = std::ceil(n * local_q - local_max_nu);
            const int m_upper = std::floor(n * local_q + local_max_nu);
            m_ranges[i].emplace_back(m_lower, m_upper);

            for (int m = m_lower; m <= m_upper; ++m) {
                const double kp = n * local_q - m;
                auto it = std::lower_bound(
                    local_omega_nu.begin(), local_omega_nu.end(), std::abs(kp),
                    [](const auto& omega_nu, auto k) {
                        return omega_nu.second.real() < k;
                    });
                if (kp > 0. &&
                    std::abs(2 * kp - std::round(2 * kp)) < EPSILON) {
                    // accumulation point, belongs to nq-m>0 branch
                    it = std::upper_bound(local_omega_nu.begin(),
                                          local_omega_nu.end(), std::abs(kp),
                                          [](auto k, const auto& omega_nu) {
                                              return k < omega_nu.second.real();
                                          });
                }
                if (it == local_omega_nu.begin()) {
                    continuum[i].push_back(0.);
                } else if (it != local_omega_nu.end()) {
                    const auto [omega0, nu0] = *(it - 1);
                    const auto [omega1, nu1] = *it;
                    continuum[i].push_back(
                        (omega0 + (std::abs(kp) - nu0.real()) /
                                      (nu1.real() - nu0.real()) *
                                      (omega1 - omega0)) /
                        local_q);
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
