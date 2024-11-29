#include <algorithm>  // lower_bound, upper_bound, max_element
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
int main(int argc, char** argv) {
    using namespace std::complex_literals;

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
    const double max_omega = 1.2;
    const int max_continuum_zone = 3;
    const bool omega_limit_by_value = true;

    const NumericEquilibrium<double> equilibrium(
        gfile_data, radial_count, poloidal_sample_point, psi_ratio);

    const auto [psi_min, psi_max] = equilibrium.psi_range();
    double q_min = std::numeric_limits<double>::infinity();
    // position of local minimum of q
    std::vector<double> q_local_extrema_pos;
    const auto delta_psi = (psi_max - psi_min) / radial_count;
    for (double psi = psi_min; psi < psi_max; psi += delta_psi) {
        auto q = equilibrium.safety_factor(psi);
        q_min = std::min(q_min, q);
        if ((q - equilibrium.safety_factor(psi - delta_psi)) *
                (q - equilibrium.safety_factor(psi + delta_psi)) >
            0) {
            q_local_extrema_pos.push_back(psi);
        }
    }
    if (q_local_extrema_pos.size() > 10) {
        std::cout << "Safety factor profile is pathological.";
        exit(1);
    }

    // toroidal mode numbers
    std::vector<int> ns{8};
    // poloidal mode numbers
    std::vector<std::vector<std::pair<int, int>>> m_ranges;
    std::vector<std::vector<double>> continuum;

    const auto n_max = *std::max_element(ns.begin(), ns.end());
    constexpr int pt_per_radial_period = 15;

    std::size_t floquet_exponent_sample_pts = 0;
    std::vector<double> psi_sample_pts;
    auto zone_iter = q_local_extrema_pos.begin();
    auto psi_left = psi_min;
    auto psi_right =
        zone_iter == q_local_extrema_pos.end()
            ? psi_max
            : (*zone_iter - psi_min < EPSILON
                   ? ++zone_iter == q_local_extrema_pos.end() ? psi_max
                                                              : *zone_iter
                   : *zone_iter);
    const auto get_next_psi = [&psi_left, &psi_right, &zone_iter,
                               &q_local_extrema_pos, &eq = equilibrium, psi_min,
                               psi_max](double psi_0, double q_diff) {
        const auto q_left = eq.safety_factor(psi_left);
        const auto q_right = eq.safety_factor(psi_right);
        auto next_q =
            eq.safety_factor(psi_0) + std::copysign(q_diff, q_right - q_left);
        const auto max_delta_psi = .01 * (psi_max - psi_min);
        if ((next_q - q_left) * (next_q - q_right) > 0) {
            if (zone_iter == q_local_extrema_pos.end()) {
                // reach right boundary
                return psi_max;
            }
            if (psi_right - psi_0 > max_delta_psi) {
                return psi_0 + max_delta_psi;
            }
            // next monotone zone
            next_q = eq.safety_factor(*zone_iter);
            psi_left = psi_right;
            psi_right =
                ++zone_iter == q_local_extrema_pos.end() ? psi_max : *zone_iter;
            return psi_left;
        }
        // find next psi according to difference in q, capped by 1% of total psi
        // to avoid points being sparse near local extrema of q
        return std::min(util::find_root(
                            [&eq, next_q](double p) {
                                return eq.safety_factor(p) - next_q;
                            },
                            psi_left, psi_right),
                        psi_0 + max_delta_psi);
    };
    for (double psi = psi_min; psi < psi_max;
         psi = get_next_psi(psi, 1. / (n_max * pt_per_radial_period))) {
        timer.pause_last_and_start_next("Calculate Floquet exponent");

        const auto local_q = equilibrium.safety_factor(psi);

        // convert between global and local normalization of $\omega$
        const auto max_local_omega = local_q / q_min * max_omega;
        auto calc_floquet_exp = [psi, &equilibrium](auto omega) {
            const auto omega2 = omega * omega;
            const auto potential = [omega2, psi,
                                    &eq = equilibrium](double theta) {
                return eq.radial_func(psi, theta) +
                       omega2 * eq.j_func(psi, theta);
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

        std::size_t order = 0;
        bool increasing = true;
        bool finish_calc_nu = false;
        double last_real = -std::numeric_limits<double>::infinity();
        std::vector<decltype(omega_nu_map)::value_type> local_omega_nu;
        for (int region = 0; !finish_calc_nu; ++region) {
            const auto omega_min =
                static_cast<double>(region) / initial_subdivision;
            const auto omega_max =
                omega_limit_by_value &&
                        region + 1 > max_local_omega * initial_subdivision
                    ? max_local_omega
                    : static_cast<double>(region + 1) / initial_subdivision;

            const auto region_begin =
                omega_nu_map.emplace(omega_min, calc_floquet_exp(omega_min))
                    .first;
            const auto region_end =
                omega_nu_map.emplace(omega_max, calc_floquet_exp(omega_max))
                    .first;
            region_stack.push({region_begin, region_end});

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
                // NOTE: I don not care about imaginary part of \nu, so points
                // in gap zone will be sparse. Extra subdivisions are done at
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
            // stopping criteria using absolute value
            finish_calc_nu =
                omega_limit_by_value &&
                region + 1 >= max_local_omega * initial_subdivision;

            // adjust $\Re\nu$ according to stability region
            for (auto it = region_begin; it != region_end; ++it) {
                auto nu = it->second;
                // normally $\Re\nu$ growth monotonic with $\omega$, but it goes
                // unchanged inside coutinuum gap, a small margin is added to
                // avoid misclassifying gap region as another stability
                // region
                if (increasing && nu.real() - last_real + EPSILON < 0. &&
                    last_real > .4) {
                    // e^{i\nu T} entering lower half plane
                    increasing = false;
                    ++order;
                } else if (!increasing &&
                           nu.real() - last_real - EPSILON > 0. &&
                           last_real < .1) {
                    // e^{i\nu T} entering upper half plane
                    increasing = true;
                    ++order;
                }

                // stopping criteria using continuum zone
                if (!omega_limit_by_value &&
                    (order == max_continuum_zone - 1 &&
                         (nu.real() < EPSILON || .5 - nu.real() < EPSILON) ||
                     order == max_continuum_zone)) {
                    // end outer loop too
                    finish_calc_nu = true;
                    break;
                }

                last_real = nu.real();
                nu.real(.5 * static_cast<double>(order) +
                        (order % 2 == 0 ? last_real : .5 - last_real));
                local_omega_nu.emplace_back(it->first, nu);
            }
        }

        timer.pause_last_and_start_next("Solve for omega");

        std::cout << "psi/psi_w = " << std::fixed << std::setprecision(4)
                  << psi / psi_max
                  << ", omega sample pt = " << local_omega_nu.size() << '\n';
        floquet_exponent_sample_pts += local_omega_nu.size();

        m_ranges.emplace_back();
        continuum.emplace_back();

        // calculate omega for each pair of mode numbers (n, m)
        // change normalization of omega to v_{A,0}/R_0 here
        const auto local_max_nu = local_omega_nu.back().second.real();
        for (std::size_t n_idx = 0; n_idx < ns.size(); ++n_idx) {
            auto n = ns[n_idx];
            std::size_t m_num = 0;
            const int m_lower = std::ceil(n * local_q - local_max_nu);
            const int m_upper = std::floor(n * local_q + local_max_nu);
            m_ranges.back().emplace_back(m_lower, m_upper);

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
                    continuum.back().push_back(0.);
                } else if (it != local_omega_nu.end()) {
                    const auto [omega0, nu0] = *(it - 1);
                    const auto [omega1, nu1] = *it;
                    continuum.back().push_back(
                        (omega0 + (std::abs(kp) - nu0.real()) /
                                      (nu1.real() - nu0.real()) *
                                      (omega1 - omega0)) /
                        local_q);
                }
            }
        }
        psi_sample_pts.push_back(psi);
    }

    std::cout << "Samples " << psi_sample_pts.size()
              << " points along radial direction.\n"
              << "Calculating Fluoquet exponent for "
              << floquet_exponent_sample_pts << " (r, omega) points.\n";

    timer.pause_last_and_start_next("Sort points into lines");

    // sort by (n,m) or (n, \nu) to individual lines
    std::unordered_map<std::pair<int, int>, std::vector<std::array<double, 2>>>
        lines;
    const bool sort_by_m = false;
    if (sort_by_m) {
        // To be deprecated
        for (std::size_t i = 0; i < psi_sample_pts.size(); ++i) {
            auto psi = psi_sample_pts[i];
            std::size_t offset = 0;
            for (std::size_t j = 0; j < ns.size(); ++j) {
                auto [m_lower, m_upper] = m_ranges[i][j];
                for (int k = 0; k <= m_upper - m_lower; ++k) {
                    lines[std::make_pair(ns[j], k + m_lower)].push_back(
                        {equilibrium.minor_radius(psi),
                         continuum[i][k + offset]});
                }
                offset += m_upper - m_lower + 1;
            }
        }
    } else {
        // sort by Floquet exponent
        for (std::size_t i = 0; i < psi_sample_pts.size(); ++i) {
            auto psi = psi_sample_pts[i];
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
