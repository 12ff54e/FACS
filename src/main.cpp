#include <algorithm>   // lower_bound, upper_bound, max_element
#include <cmath>       // ceil, floor
#include <cstdint>     // int32_t
#include <cstdlib>     // exit
#include <filesystem>  // path
#include <iostream>
#include <limits>  // infinity
#include <map>
#include <numbers>  // pi
#include <sstream>
#include <stack>
#include <unordered_map>

#include "Floquet.h"
#include "clap.h"
#include "equilibrium.h"
#include "integrator.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/console.h>  // emscripten_console_log
#include <emscripten/wasm_worker.h>

// lock for m_ranges and continuum vector
emscripten_lock_t lock = EMSCRIPTEN_LOCK_T_STATIC_INITIALIZER;
// TODO: Add axis in preview
#endif

#define ZQ_TIMER_IMPLEMENTATION
#include "timer.h"

constexpr std::size_t radial_sample_point = 1024;
constexpr std::size_t poloidal_sample_point = 256;

// inject hash function for pair of int
namespace std {
template <>
struct hash<pair<int, int>> {
    auto operator()(const pair<int, int>& p) const noexcept {
        return hash<int>{}(p.first) ^ (hash<int>{}(p.second) << 1);
    }
};
};  // namespace std

struct Input {
    int max_continuum_zone;
    int radial_grid_num;
    double max_omega_value;
    double psi_ratio;
    std::string gfile_path;
    std::string output_path;
    std::string output_nu_path;
    std::string toroidal_mode_num_str;

    bool omega_limit_by_value() const { return max_continuum_zone == 0; }
    bool adaptive_radial_grid() const { return radial_grid_num == 0; }
};

// every member should be zero-initialized, which is exactly the behavior of
// that of static storage duration
struct State {
#ifdef __EMSCRIPTEN__
    // flow control
    bool finish_calculation;
    bool stats_printed;

    // for real time preview
    int last_radial_idx;
    int current_radial_idx;
    int total_offset;
    std::vector<double> r_sample_pts;

    // worker state
    emscripten_wasm_worker_t worker_id;
#endif
    // input
    Input input;
    std::string gfile_content;
    std::vector<int> ns;

    // intermediates
    std::vector<double> psi_sample_pts;
    std::vector<int32_t> m_ranges;
    std::vector<double> continuum;

    // result
    std::unordered_map<std::pair<int, int>, std::vector<std::array<double, 2>>>
        lines;

    // extra info
    double psi_max;
};

auto& get_state() {
    static State state;
    return state;
}

// Zero criteria for float point numbers
constexpr double EPSILON = 1.e-6;

void calculate_continuum(const auto& equilibrium) {
    using namespace std::complex_literals;
    auto& timer = Timer::get_timer();

    const auto [psi_min, psi_max_eq] = equilibrium.psi_range();
    const auto psi_max = psi_max_eq * get_state().input.psi_ratio;

    // position of local extrema of q
    std::vector<double> q_local_extrema_pos;
    const auto delta_psi = (psi_max - psi_min) / radial_sample_point;
    double q_min_psi;
    double q_min = std::numeric_limits<double>::infinity();
    for (double psi = psi_min; psi < psi_max; psi += delta_psi) {
        auto q = equilibrium.safety_factor(psi);
        if (q_min > q) {
            q_min = q;
            q_min_psi = psi;
        }
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

    auto& psi_sample_pts = get_state().psi_sample_pts;

    const auto& ns = get_state().ns;
    const auto n_max = *std::max_element(ns.begin(), ns.end());
    constexpr int pt_per_radial_period = 15;

    std::size_t floquet_exponent_sample_pts = 0;
    auto zone_iter = q_local_extrema_pos.begin();
    auto psi_left = decltype(psi_min){};
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

    auto calculate_local_floquet_func = [&equilibrium](auto psi) {
        const auto local_q = equilibrium.safety_factor(psi);
        const auto max_continuum_zone = get_state().input.max_continuum_zone;
        const auto omega_limit_by_value =
            get_state().input.omega_limit_by_value();

        // convert between global and local normalization of $\omega$
        const auto max_local_omega =
            local_q * get_state().input.max_omega_value;
        auto calc_floquet_exp = [psi, &equilibrium](auto omega) {
            const auto omega2 = omega * omega;
            const auto potential = [omega2, psi,
                                    &eq = equilibrium](double theta) {
                return eq.radial_func(psi, theta) +
                       omega2 * eq.j_func(psi, theta);
            };
            const FloquetMatrix flo_mat(
                potential, std::numbers::pi * 2,
                static_cast<std::size_t>(100 * omega * 2) + 100);
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
                // NOTE: I don not care about imaginary part of \nu, so
                // points in gap zone will be sparse. Extra subdivisions are
                // done at gap-continuum boundary.
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
                // normally $\Re\nu$ growth monotonic with $\omega$, but it
                // goes unchanged inside coutinuum gap, a small margin is
                // added to avoid misclassifying gap region as another
                // stability region
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

                last_real = nu.real();
                nu.real(.5 * static_cast<double>(order) +
                        (order % 2 == 0 ? last_real : .5 - last_real));
                local_omega_nu.emplace_back(it->first, nu);

                // stopping criteria using continuum zone
                if (!omega_limit_by_value &&
                    (order == max_continuum_zone - 1 &&
                         (.5 * static_cast<double>(max_continuum_zone) -
                              nu.real() <
                          EPSILON) ||
                     order == max_continuum_zone)) {
                    // end outer loop too
                    finish_calc_nu = true;
                    break;
                }
            }
        }

        return local_omega_nu;
    };

    const auto calculate_omega = [](const auto& local_omega_nu, auto local_q) {
        const auto& ns = get_state().ns;
        const auto local_max_nu = local_omega_nu.back().second.real();
        auto& m_ranges = get_state().m_ranges;
        auto& continuum = get_state().continuum;
        for (std::size_t n_idx = 0; n_idx < ns.size(); ++n_idx) {
            auto n = ns[n_idx];
            std::size_t m_num = 0;
            const int m_lower = std::ceil(n * local_q - local_max_nu);
            const int m_upper = std::floor(n * local_q + local_max_nu);
            m_ranges.push_back(m_lower);
            m_ranges.push_back(m_upper);

            for (int m = m_lower; m <= m_upper; ++m) {
                const double kp = n * local_q - m;
                auto it = std::lower_bound(
                    local_omega_nu.begin(), local_omega_nu.end(), std::abs(kp),
                    [](const auto& omega_nu, auto k) {
                        // NOTE: Here I presume omega being half integer
                        // lies inside gap region, which should be fine
                        // providing that radial metric have tiny dc
                        // component.
                        return omega_nu.second.real() < k ||
                               omega_nu.first < .5 * std::floor(2 * k);
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
                    continuum.push_back(0.);
                } else if (it != local_omega_nu.end()) {
                    const auto [omega0, nu0] = *(it - 1);
                    const auto [omega1, nu1] = *it;
                    // change normalization of omega to v_{A,0}/R_0 here
                    continuum.push_back(
                        (omega0 + (std::abs(kp) - nu0.real()) /
                                      (nu1.real() - nu0.real()) *
                                      (omega1 - omega0)) /
                        local_q);
                }
            }
        }
    };
#ifdef __EMSCRIPTEN__
    if (!get_state().input.omega_limit_by_value()) {
        // estimate max omega, and inform js for preview purpose
        auto omega_nu = calculate_local_floquet_func(q_min_psi);
        emscripten_wasm_worker_post_function_vd(
            EMSCRIPTEN_WASM_WORKER_ID_PARENT,
            [](double omega) { EM_ASM({ set_max_omega($0); }, omega); },
            omega_nu.back().first / q_min);
    }
#endif

    auto get_psi_evenly = [n = get_state().input.radial_grid_num,
                           psi_max](auto idx) {
        return std::pow(static_cast<double>(idx) / n, 2) * psi_max;
    };
#ifndef __EMSCRIPTEN__
    std::ofstream nu_ofs;
    if (get_state().input.output_nu_path.size() > 0) {
        nu_ofs.open(get_state().input.output_nu_path);
    }
#endif
    // TODO: Equilibrium value near magnetic axis is calculated by linear
    // interpolation, which might be awfully inaccurate. Consider using Miller
    // model to approximate equlibrium
    std::size_t idx = 0;
    for (double psi = 0; psi < psi_max;
         psi = get_state().input.adaptive_radial_grid()
                   ? get_next_psi(psi, 1. / (n_max * pt_per_radial_period))
                   : get_psi_evenly(idx++)) {
        const auto local_q = equilibrium.safety_factor(psi);
        timer.pause_last_and_start_next("Calculate Floquet exponent");
        auto local_omega_nu = calculate_local_floquet_func(psi);
#ifndef __EMSCRIPTEN__
        if (nu_ofs.is_open()) {
            nu_ofs << local_omega_nu.size() << ' '
                   << equilibrium.minor_radius(psi);
            for (const auto [omega, nu] : local_omega_nu) {
                nu_ofs << ' ' << omega / local_q << ' ' << nu.real() << ' '
                       << nu.imag();
            }
            nu_ofs << '\n';
        }
#endif
        timer.pause_last_and_start_next("Solve for omega");
        {
            std::ostringstream oss;
            oss << "psi/psi_w = " << std::fixed << std::setprecision(4)
                << psi / psi_max_eq
                << ", omega sample pt = " << local_omega_nu.size() << '\n';
#ifdef __EMSCRIPTEN__
            emscripten_console_log(oss.str().c_str());
#else
            std::cout << oss.str();
#endif
        }
        floquet_exponent_sample_pts += local_omega_nu.size();

#ifdef __EMSCRIPTEN__
        // This lock is necessary since vector might be moved (through memmove
        // syscall) when growing
        emscripten_lock_waitinf_acquire(&lock);
#endif
        // calculate omega for each pair of mode numbers (n, m)
        calculate_omega(local_omega_nu, local_q);

#ifdef __EMSCRIPTEN__
        emscripten_lock_release(&lock);
        ++get_state().current_radial_idx;
        get_state().r_sample_pts.push_back(equilibrium.minor_radius(psi));
#endif
        psi_sample_pts.push_back(psi);
    }
#ifndef __EMSCRIPTEN__
    if (nu_ofs.is_open()) { nu_ofs.close(); };
#endif
    {
        std::ostringstream oss;
        oss << "Samples " << psi_sample_pts.size()
            << " points along radial direction.\n"
            << "Calculating Fluoquet exponent for "
            << floquet_exponent_sample_pts << " (r, omega) points.\n";
#ifdef __EMSCRIPTEN__
        get_state().finish_calculation = true;  // mark finish
        emscripten_console_log(oss.str().c_str());
#else
        std::cout << oss.str();
#endif
    }
}

auto sort_points_into_lines(const auto& equilibrium) {
    // sort by (n,m) or (n, \nu) to individual lines
    std::unordered_map<std::pair<int, int>, std::vector<std::array<double, 2>>>
        lines;
    const bool sort_by_m = false;
    const auto& ns = get_state().ns;
    const auto& psi_sample_pts = get_state().psi_sample_pts;
    const auto& m_ranges = get_state().m_ranges;
    // sort by Floquet exponent
    std::size_t cont_idx = 0;
    std::size_t m_idx = 0;
    for (auto psi : psi_sample_pts) {
        const auto minor_radius = equilibrium.minor_radius(psi);
        const auto q = equilibrium.safety_factor(psi);
        for (std::size_t j = 0; j < ns.size(); ++j) {
            auto m_lower = m_ranges[m_idx++];
            auto m_upper = m_ranges[m_idx++];
            for (int k = 0; k <= m_upper - m_lower; ++k) {
                const int kp = std::floor(
                    std::abs(.5 + std::floor(2 * (ns[j] * q - (k + m_lower)))));
                lines[std::make_pair(ns[j], sort_by_m ? k + m_lower : kp)]
                    .push_back(
                        {minor_radius, get_state().continuum[cont_idx++]});
            }
        }
    }

    return lines;
}

void gfile_to_continuum_lines() {
    auto& timer = Timer::get_timer();
    const auto& input = get_state().input;

    timer.start("Construct Equilibrium");
    std::istringstream gfile_stream(get_state().gfile_content);
    GFileRawData gfile_data;
    gfile_stream >> gfile_data;
    if (!gfile_data.is_complete()) {
        std::cerr << "Can not parse g-file.\n";
        std::exit(0);
    }

    const NumericEquilibrium<double> equilibrium(
        gfile_data, radial_sample_point, poloidal_sample_point);
    get_state().psi_max = equilibrium.psi_range().second;

    timer.pause();

    calculate_continuum(equilibrium);

    timer.pause_last_and_start_next("Sort points into lines");

    // sort by (n,m) or (n, \nu) to individual lines
    get_state().lines = sort_points_into_lines(equilibrium);
    timer.pause();
};

void output() {
    auto& timer = Timer::get_timer();
    timer.start("Output");

    const auto& output_path = get_state().input.output_path;
    std::ofstream output(output_path);
    if (!output.is_open()) {
        std::cerr << "Failed to open " << std::quoted(output_path)
                  << " for write.";
        std::exit(ENOENT);
    }
    output << get_state().psi_max << '\n';
    for (auto& line : get_state().lines) {
        const auto& [nm, coords] = line;
        output << nm.first << ' ' << nm.second << ' ';
        for (auto pt : coords) { output << pt[0] << ' ' << pt[1] << ' '; }
        output << '\n';
    }

    output.close();

    timer.pause();
    timer.print();
}

#ifdef __EMSCRIPTEN__
char worker_stack[8192];

void draw_pts() {
    const auto current_radial_idx = get_state().current_radial_idx;
    auto& last_radial_idx = get_state().last_radial_idx;
    const auto& continuum = get_state().continuum;
    const auto& m_ranges = get_state().m_ranges;
    const auto n_size = get_state().ns.size();

    if (current_radial_idx > last_radial_idx) {
        emscripten_lock_busyspin_waitinf_acquire(&lock);
        auto& total_offset = get_state().total_offset;
        for (int i = last_radial_idx; i < current_radial_idx; ++i) {
            total_offset += EM_ASM_INT(
                { return draw_point($0, $1, $2, $3); },
                get_state().r_sample_pts[i], m_ranges.data() + 2 * i * n_size,
                2 * n_size, continuum.data() + total_offset);
        }
        emscripten_lock_release(&lock);
    }
    last_radial_idx = current_radial_idx;
    if (get_state().finish_calculation && !get_state().stats_printed) {
        output();
        get_state().stats_printed = true;

        EM_ASM({ enable_download(); });

        emscripten_terminate_wasm_worker(get_state().worker_id);
    }
}
#endif

int main(int argc, char** argv) {
#ifdef __EMSCRIPTEN__
    // clear state since main might be called multiple times
    get_state() = {};

    Timer::get_timer().reset();
#endif

    CLAP_BEGIN(Input)
    CLAP_ADD_USAGE("[OPTION]... INPUT_FILE")
    CLAP_ADD_DESCRIPTION(
        "Calculate Alfvenic continuum of the given equilibrium.")
    CLAP_REGISTER_ARG(gfile_path)
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        max_continuum_zone, "--max-continuum-zone", "-m",
        "Specify the maximum continuum zone for omega. For example, set it to "
        "2 if you just want to check the TAE gap.")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        max_omega_value, "--max-value",
        "Specify maximum omega value, in the unit of v_{A,0}/R_0.")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        output_path, "--output-path", "-o",
        "Specify the path of output file, default to '$PWD/continuum-${input "
        "file name}'")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        output_nu_path, "--output-floquet-exponent-to",
        "Specify the path of output file of Floquet exponent")

    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        radial_grid_num, "--radial-grid-num",
        "Number of radial grid. Radial grid will be determined adaptively if "
        "this value is not given.")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        toroidal_mode_num_str, "--toroidal-mode-number", "-n",
        "Toroidal mode number list, should be seperated by comma.")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        psi_ratio, "--psi-ratio",
        "Maximum poloidal flux relative to that on LCFS, default to be 1")
    CLAP_END(Input)

    auto& input = get_state().input;
    input.psi_ratio = 1.;

    try {
        CLAP<Input>::parse_input(input, argc, argv);
    } catch (std::exception& e) {
        std::cout << e.what();
        return EINVAL;
    }

    // check input options

    if (input.gfile_path.empty()) {
        std::cerr << "Please provide input file.\n";
        return ENOENT;
    }
    if (input.toroidal_mode_num_str.empty()) {
        std::cout << "User do not specify toroidal mode number, use `-n 3,5` "
                     "as default";
        input.toroidal_mode_num_str = "3,5";
    }
    // parse toroidal mode numbers
    if (([](const auto& str, auto& ns) {
            int idx = 0;
            do {
                int n = std::atoi(str.c_str() + idx);
                if (n == 0) { return true; }
                ns.push_back(n);

                idx = str.find(',', idx);
            } while (idx != str.npos && ++idx != str.size());
            return false;
        })(input.toroidal_mode_num_str, get_state().ns)) {
        std::cout << "Can not recognize the given toroidal mode number(s).\n";
        return EINVAL;
    }

    auto output_path = std::filesystem::path{input.output_path};
    if (!output_path.has_filename()) {
        const auto output_filename =
            std::string{"continuum-"} +
            std::filesystem::path{input.gfile_path}.filename().string();
        output_path.replace_filename(output_filename);
        input.output_path = output_path.string();
    }
    auto output_nu_path = std::filesystem::path{input.output_nu_path};
    if (!output_nu_path.empty() && !output_nu_path.has_filename()) {
        const auto output_filename =
            std::string{"floquet-exponent-"} +
            std::filesystem::path{input.gfile_path}.filename().string();
        output_nu_path.replace_filename(output_filename);
        input.output_nu_path = output_nu_path.string();
    }

    if (input.max_omega_value == 0 && input.max_continuum_zone == 0) {
        std::cout << "User do no specify maximum value of omega, use `-m 2` as "
                     "default.\n";
        input.max_continuum_zone = 2;
    }

    auto& timer = Timer::get_timer();
    timer.start("Read gfile");

    std::ifstream gfile(input.gfile_path);
    if (!gfile.is_open()) {
        std::cerr << "Can not open g-file.\n";
        return ENOENT;
    }
    std::ostringstream buffer;
    buffer << gfile.rdbuf();
    get_state().gfile_content = buffer.str();
    gfile.close();

    timer.pause();

#ifdef __EMSCRIPTEN__  // delegates to wasm worker
    auto wasm_worker =
        emscripten_create_wasm_worker(worker_stack, sizeof(worker_stack));

    if (wasm_worker == 0) {
        std::cout << "Failed to create Wasm worker.\n";
        return 1;
    }
    get_state().worker_id = wasm_worker;
    emscripten_wasm_worker_post_function_v(wasm_worker,
                                           gfile_to_continuum_lines);

    if (input.omega_limit_by_value()) {
        EM_ASM({set_max_omega($0)}, input.max_omega_value);
    }
    emscripten_set_main_loop(draw_pts, 0, false);
#else  // perform calculation in main thread
    gfile_to_continuum_lines();
    output();
#endif

    return 0;
}
