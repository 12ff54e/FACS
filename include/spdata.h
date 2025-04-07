#ifndef SPDATA_H
#define SPDATA_H

#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>

#ifdef __EMSCRIPTEN__
#include <emscripten/console.h>
#endif

#include "contour.h"
#include "gFileRawData.h"
#include "intp.h"

template <typename T>
class Spdata {
   private:
    // interpolation order of internal use
    constexpr static std::size_t ORDER_ = 3;
    constexpr static std::size_t ORDER_OUT_ = 3;

   public:
    using value_type = T;
    using coord_type = double;

    static constexpr std::size_t FIELD_NUM_2D = 5;
    static constexpr std::size_t FIELD_NUM_1D = 6;

    // The contour mesh grid of data stored here, and the resolution should be
    // higher than the output spec
    struct SpdataRaw_ {
        // - magnetic_field
        // - r
        // - z
        // - jacobian
        // - \sqrt{g^{rr}}^{-1} \partial^2_\theta \sqrt{g^{rr}}
        std::array<intp::Mesh<value_type, 2>, FIELD_NUM_2D> data_2d;
        // - safety_factor
        // - poloidal_current
        // - toroidal_current
        // - pressure
        // - minor_radius
        // - toroidal_flux
        std::array<std::vector<value_type>, FIELD_NUM_1D> data_1d;

        std::array<value_type, FIELD_NUM_2D> axis_value_2d;
        std::array<value_type, FIELD_NUM_1D> axis_value_1d;

        value_type flux_unit;
    };

    struct SpdataIntp_ {
        std::vector<value_type> psi_sample_for_output;

        // - magnetic_field
        // - r
        // - z
        // - jacobian
        // - \sqrt{g^{rr}}^{-1} \partial^2_\theta \sqrt{g^{rr}}
        std::array<
            intp::InterpolationFunction<value_type, 2u, ORDER_OUT_, coord_type>,
            FIELD_NUM_2D>
            intp_2d;
        // - safety_factor
        // - poloidal_current
        // - toroidal_current
        // - pressure
        // - minor_radius
        // - toroidal_flux
        std::array<
            intp::InterpolationFunction1D<ORDER_OUT_, value_type, coord_type>,
            FIELD_NUM_1D>
            intp_1d;

        template <std::size_t... indices_2d, std::size_t... indices_1d>
        SpdataIntp_(const SpdataRaw_& spdata_raw,
                    const Spdata& spdata,
                    std::index_sequence<indices_2d...>,
                    std::index_sequence<indices_1d...>)
            : psi_sample_for_output{spdata.generate_psi_sample_for_output_(
                  spdata_raw.flux_unit)},
              intp_2d{spdata.create_2d_spline_(spdata_raw.data_2d[indices_2d],
                                               psi_sample_for_output)...},
              intp_1d{spdata.create_1d_spline_(spdata_raw.data_1d[indices_1d],
                                               psi_sample_for_output)...} {}
    };

    Spdata(const GFileRawData& g_file_data,
           std::size_t radial_grid_num,
           std::size_t poloidal_grid_num,
           bool use_si = false,
           std::size_t radial_sample = 256,
           value_type psi_ratio = .99)
        : use_si_(use_si),
          lsp_(radial_grid_num),
          lst_(poloidal_grid_num),
          psi_delta_(psi_ratio *
                     (g_file_data.flux_LCFS - g_file_data.flux_magnetic_axis) /
                     static_cast<value_type>(lsp_ - 1)),
          theta_delta_(2. * M_PI / static_cast<value_type>(lst_)),
          spdata_raw_{generate_boozer_coordinate_(g_file_data,
                                                  radial_sample,
                                                  psi_ratio)},
          spdata_intp_{spdata_raw_, *this,
                       std::make_index_sequence<FIELD_NUM_2D>{},
                       std::make_index_sequence<FIELD_NUM_1D>{}} {}

    const auto& intp_data() const { return spdata_intp_; }

    const auto radial_grid_num() const { return lsp_; }
    const auto poloidal_grid_num() const { return lst_; }

   protected:
    const bool use_si_;
    const std::size_t lsp_, lst_;
    value_type psi_delta_;
    const value_type theta_delta_;
    const SpdataRaw_ spdata_raw_;
    const SpdataIntp_ spdata_intp_;

    SpdataRaw_ generate_boozer_coordinate_(const GFileRawData& g_file_data,
                                           std::size_t radial_sample,
                                           value_type psi_ratio) {
        intp::InterpolationFunction<value_type, 2u, ORDER_, coord_type>
            flux_function(
                g_file_data.flux,
                std::make_pair(g_file_data.r_left,
                               g_file_data.r_left + g_file_data.dim.x()),
                std::make_pair(g_file_data.z_mid - .5 * g_file_data.dim.y(),
                               g_file_data.z_mid + .5 * g_file_data.dim.y()));

        // check poloidal flux value at magnetic axis
        const auto psi_ma_intp = flux_function(g_file_data.magnetic_axis);
        if (std::abs((psi_ma_intp - g_file_data.flux_magnetic_axis) /
                     (g_file_data.flux_LCFS - g_file_data.flux_magnetic_axis)) >
            1.e-4) {
            std::ostringstream oss;
            oss << "The poloidal flux of magnetic axis given in gfile "
                   "deviates from interpolated value.\n"
                << "  \\psi_p in gfile: " << g_file_data.flux_magnetic_axis
                << "\n  \\psi_p from interpolation: " << psi_ma_intp << '\n';
#ifdef __EMSCRIPTEN__
            emscripten_console_log(oss.str().c_str());
#else
            std::cout << oss.str();
#endif
        }

        value_type flux_boundary_min = 10. * std::pow(g_file_data.r_center, 2) *
                                       std::abs(g_file_data.b_center);
        for (const auto& pt : g_file_data.boundary) {
            flux_boundary_min = std::min(flux_boundary_min, flux_function(pt));
        }
        {
            std::ostringstream oss;
            oss << "The poloidal flux of last closed flux surface is "
                << g_file_data.flux_LCFS << '\n'
                << "Minimum of interpolated value at boundary points is "
                << flux_boundary_min << '\n';
#ifdef __EMSCRIPTEN__
            emscripten_console_log(oss.str().c_str());
#else
            std::cout << oss.str();
#endif
        }
        const value_type psi_bd = flux_boundary_min - psi_ma_intp;
        const value_type psi_wall = psi_ratio * psi_bd;
        psi_delta_ = psi_wall / static_cast<value_type>(lsp_ - 1);

        // contours are from \\Delta\\psi to LCFS
        std::vector<Contour<value_type>> contours;
        contours.reserve(radial_sample);
        for (std::size_t i = 0; i < radial_sample; ++i) {
            const auto psi =
                i == radial_sample - 1
                    ? flux_boundary_min
                    : util::lerp(
                          psi_delta_, psi_wall,
                          static_cast<value_type>(i) /
                              static_cast<value_type>(radial_sample - 1)) +
                          psi_ma_intp;
            contours.emplace_back(psi, flux_function, g_file_data);
        }

        constexpr value_type magnetic_constant = 4.e-7 * M_PI;

        // safety factor, on shifted psi
        intp::InterpolationFunction1D<ORDER_, value_type, coord_type>
            safety_factor_intp{
                std::make_pair(0., psi_bd),
                intp::util::get_range(g_file_data.safety_factor),
            };

        // poloidal current, on shifted psi
        intp::InterpolationFunction1D<ORDER_, value_type, coord_type>
            poloidal_current_intp{
                std::make_pair(0., psi_bd),
                intp::util::get_range(g_file_data.f_pol),
            };

        // pressure, on shifted psi
        intp::InterpolationFunction1D<ORDER_, value_type, coord_type>
            pressure_intp{std::make_pair(0., psi_bd),
                          intp::util::get_range(g_file_data.pressure)};

        // this following function accepts shifted psi (0 at m.a.)

        auto b2j_field = [&](Vec<2, value_type> pt, value_type psi) {
            value_type dp_dr = flux_function.derivative(pt, {1, 0});
            value_type dp_dz = flux_function.derivative(pt, {0, 1});
            value_type r = pt.x();
            pt -= g_file_data.magnetic_axis;
            value_type r2 = pt.L2_norm_square_();
            value_type f = poloidal_current_intp(psi);

            return (f * f + dp_dr * dp_dr + dp_dz * dp_dz) * r2 /
                   (r * (dp_dr * pt.x() + dp_dz * pt.y()));
        };

        auto b_field = [&](Vec<2, value_type> pt, value_type psi) {
            value_type dp_dr = flux_function.derivative(pt, {1, 0});
            value_type dp_dz = flux_function.derivative(pt, {0, 1});
            value_type f = poloidal_current_intp(psi);

            return std::sqrt(f * f + dp_dr * dp_dr + dp_dz * dp_dz) / pt.x();
        };

        auto bp2j_field = [&](Vec<2, value_type> pt, value_type) {
            value_type dp_dr = flux_function.derivative(pt, {1, 0});
            value_type dp_dz = flux_function.derivative(pt, {0, 1});
            value_type r = pt.x();
            pt -= g_file_data.magnetic_axis;
            value_type r2 = pt.L2_norm_square_();

            return (dp_dr * dp_dr + dp_dz * dp_dz) * r2 /
                   ((dp_dr * pt.x() + dp_dz * pt.y()) * r);
        };

        auto gp_rt = [&](Vec<2, value_type> pt, value_type) {
            auto dp_dr = flux_function.derivative(pt, {1, 0});
            auto dp_dz = flux_function.derivative(pt, {0, 1});
            return std::sqrt(dp_dr * dp_dr + dp_dz * dp_dz);
        };

        constexpr value_type PI2 = 2 * M_PI;

        std::vector<value_type> poloidal_angles{
            g_file_data.geometric_poloidal_angles};
        poloidal_angles.push_back(poloidal_angles.front() + PI2);
        // \\theta range: \\theta_0, ..., \\theta_0 + 2\\pi
        intp::InterpolationFunctionTemplate1D<ORDER_, value_type, coord_type>
            poloidal_template{intp::util::get_range(poloidal_angles),
                              poloidal_angles.size(), true};

        if (std::fpclassify(poloidal_angles.front()) != FP_ZERO) {
            poloidal_angles.insert(poloidal_angles.begin(), 0);
        }

        poloidal_angles.back() = PI2;
        // \\theta range: 0, ..., 2\\pi
        intp::InterpolationFunctionTemplate1D<ORDER_, value_type, coord_type>
            poloidal_template_full{intp::util::get_range(poloidal_angles),
                                   poloidal_angles.size(), false};

        // output data

        intp::Mesh<value_type, 2> magnetic_boozer(radial_sample, lst_ + 1);
        intp::Mesh<value_type, 2> r_boozer(radial_sample, lst_ + 1);
        intp::Mesh<value_type, 2> z_boozer(radial_sample, lst_ + 1);
        intp::Mesh<value_type, 2> jacobian_boozer(radial_sample, lst_ + 1);
        intp::Mesh<value_type, 2> radial_func_boozer(radial_sample, lst_ + 1);

        std::vector<value_type> safety_factor, pol_current_n, tor_current_n,
            pressure_n, r_minor_n, tor_flux_n;

        safety_factor.reserve(radial_sample);
        pol_current_n.reserve(radial_sample);
        tor_current_n.reserve(radial_sample);
        pressure_n.reserve(radial_sample);
        r_minor_n.reserve(radial_sample);
        tor_flux_n.reserve(radial_sample);

        const value_type B0 = b_field(g_file_data.magnetic_axis, 0.);
        const value_type R0 = g_file_data.magnetic_axis.x();

        // This two basic unit determines the output spdata unit,
        // setting them to 1 means SI unit.
        const value_type length_unit = use_si_ ? 1. : R0;
        const value_type magnetic_field_unit = use_si_ ? 1. : B0;

        const value_type current_unit = length_unit * magnetic_field_unit;
        const value_type pressure_unit =
            magnetic_field_unit * magnetic_field_unit / magnetic_constant;
        const value_type flux_unit =
            length_unit * length_unit * magnetic_field_unit;

        // construct boozer coordinate, integrate B^2 * J along each contour

#define boozer_list() \
    X(r);             \
    X(z);             \
    X(b2j);           \
    X(bp2j);          \
    X(gp_rt);

        for (std::size_t ri = 0; ri < contours.size(); ++ri) {
            const value_type psi = contours[ri].flux() - psi_ma_intp;
            const std::size_t poloidal_size = contours[ri].size() + 1;
#define X(name)                         \
    std::vector<value_type> name##_geo; \
    name##_geo.reserve(poloidal_size)
            boozer_list();
#undef X

            // quantities on geometric grid
            for (size_t i = 0; i < poloidal_size; ++i) {
                const auto& pt = contours[ri][i % (poloidal_size - 1)];
                r_geo.push_back(pt.x());
                z_geo.push_back(pt.y());
                b2j_geo.push_back(b2j_field(pt, psi));
                bp2j_geo.push_back(bp2j_field(pt, psi));
                gp_rt_geo.push_back(gp_rt(pt, psi));
            }
            // interpolation on geometric grid
#define X(name)            \
    auto name##_geo_intp = \
        poloidal_template.interpolate(intp::util::get_range(name##_geo))
            boozer_list();
#undef X

            // integrate

            std::vector<value_type> b2j_int;
            b2j_int.reserve(poloidal_angles.size());
            b2j_int.push_back(0);

            std::vector<value_type> bp2j_int;
            bp2j_int.reserve(poloidal_angles.size());
            bp2j_int.push_back(0);

            // Poloidal grid begins from \\theta = 0 and ends at \\theta = 2\\pi
            for (size_t i = 1; i < poloidal_angles.size(); ++i) {
                b2j_int.push_back(b2j_int.back() +
                                  util::integrate_coarse(b2j_geo_intp,
                                                         poloidal_angles[i - 1],
                                                         poloidal_angles[i]));
                bp2j_int.push_back(bp2j_int.back() + util::integrate_coarse(
                                                         bp2j_geo_intp,
                                                         poloidal_angles[i - 1],
                                                         poloidal_angles[i]));
            }
            auto coef = b2j_int.back() / PI2;
            tor_current_n.push_back(bp2j_int.back() / (PI2 * current_unit));
            // normalization
            for (auto& v : b2j_int) { v /= coef; }
            auto boozer_geo_intp = poloidal_template_full.interpolate(
                intp::util::get_range(b2j_int));

            std::vector<value_type> gp_rt_boozer_grid;
            gp_rt_boozer_grid.reserve(lst_ + 1);

            // calculate necessary values on a even-spaced boozer grid
            for (size_t i = 0; i <= lst_; ++i) {
                value_type theta_boozer =
                    (static_cast<value_type>(i % lst_) + .5) * theta_delta_;
                value_type theta_geo = util::find_root(
                    [&](value_type t) {
                        return boozer_geo_intp(t) - theta_boozer;
                    },
                    0., PI2);
                value_type r_grid = r_geo_intp(theta_geo);
                value_type z_grid = z_geo_intp(theta_geo);

                // be careful of normalization

                auto b = b_field({r_grid, z_grid}, psi);

                magnetic_boozer(ri, i) = b / magnetic_field_unit;
                r_boozer(ri, i) = r_grid / length_unit;
                // z value is shifted such that magnetic axis has z = 0
                z_boozer(ri, i) =
                    (z_grid - g_file_data.magnetic_axis.y()) / length_unit;
                jacobian_boozer(ri, i) =
                    coef / (b * b) * magnetic_field_unit / length_unit;
                gp_rt_boozer_grid.push_back(gp_rt({r_grid, z_grid}, psi));
            }
            // Resample on a sparser grid to avoid a noisy 2nd order derivative.
            // Smooth before interpolation might also help, e.g. convolution
            // with a Gauss kernel.
            {
                intp::InterpolationFunction1D<ORDER_, value_type, coord_type>
                    gp_rt_intp(std::make_pair(.5 * theta_delta_,
                                              2 * M_PI + .5 * theta_delta_),
                               intp::util::get_range(gp_rt_boozer_grid), true);
                constexpr std::size_t gp_rt_lst = 50;
                constexpr double gp_rt_theta_delta = 2 * M_PI / gp_rt_lst;
                std::vector<value_type> gp_rt_boozer_grid_sparse;
                gp_rt_boozer_grid_sparse.reserve(gp_rt_lst + 1);
                for (std::size_t i = 0; i <= gp_rt_lst; ++i) {
                    gp_rt_boozer_grid_sparse.push_back(gp_rt_intp(
                        (static_cast<value_type>(i % gp_rt_lst) + .5) *
                        gp_rt_theta_delta));
                }

                intp::InterpolationFunction1D<ORDER_, value_type, coord_type>
                    gp_rt_sparse_intp(
                        std::make_pair(.5 * gp_rt_theta_delta,
                                       2 * M_PI + .5 * gp_rt_theta_delta),
                        intp::util::get_range(gp_rt_boozer_grid_sparse), true);
                for (std::size_t i = 0; i <= lst_; ++i) {
                    radial_func_boozer(ri, i) =
                        gp_rt_sparse_intp.derivative(
                            {(static_cast<value_type>(i % lst_) + .5) *
                             theta_delta_},
                            {2}) /
                        gp_rt_boozer_grid[i];
                }
            }

            safety_factor.push_back(safety_factor_intp(psi));
            pol_current_n.push_back(poloidal_current_intp(psi) / current_unit);
            pressure_n.push_back(pressure_intp(psi) / pressure_unit);
            tor_flux_n.push_back(
                (ri == 0 ? 0. : tor_flux_n.back()) +
                util::integrate_coarse(
                    safety_factor_intp,
                    ri == 0 ? 0. : (contours[ri - 1].flux() - psi_ma_intp),
                    psi) /
                    flux_unit);
            // r_minor defined as distance from magnetic axis at weak field side
            // this value is always normalized to R0
            r_minor_n.push_back(r_geo_intp(0.) / R0 - 1.);
        }
        const value_type q0 = safety_factor_intp(0);
        const value_type b0n = B0 / magnetic_field_unit;
        const value_type g0n = poloidal_current_intp(0) / current_unit;
        const value_type p0n = pressure_intp(0) / pressure_unit;
        return SpdataRaw_{
            {std::move(magnetic_boozer), std::move(r_boozer),
             std::move(z_boozer), std::move(jacobian_boozer),
             std::move(radial_func_boozer)},
            {std::move(safety_factor), std::move(pol_current_n),
             std::move(tor_current_n), std::move(pressure_n),
             std::move(r_minor_n), std::move(tor_flux_n)},
            {b0n, R0 / length_unit, 0., q0 * g0n / (b0n * b0n), 0.},
            {q0, g0n, 0., p0n, 0., 0.},
            flux_unit};
    }

    std::vector<value_type> generate_psi_sample_for_output_(
        value_type unit) const {
        std::vector<value_type> psi(lsp_);
        psi[0] = psi_delta_ / unit;
        for (std::size_t i = 1; i < lsp_ - 1; ++i) {
            psi[i] = psi_delta_ * (static_cast<value_type>(i) + .5) / unit;
        }
        psi[lsp_ - 1] = static_cast<value_type>(lsp_ - 1) * psi_delta_ / unit;

        return psi;
    }

    auto create_2d_spline_(const intp::Mesh<value_type, 2>& data,
                           const std::vector<value_type>& psi_sample) const {
        // interpolate the even-spaced data
        return intp::InterpolationFunction<value_type, 2u, ORDER_OUT_,
                                           coord_type>(
            {false, true}, data,
            std::make_pair(psi_sample.front(), psi_sample.back()),
            std::make_pair(.5 * theta_delta_, 2. * M_PI + .5 * theta_delta_));
    }

    auto create_1d_spline_(const std::vector<value_type>& data,
                           const std::vector<value_type>& psi_sample) const {
        // interpolate the even-spaced data
        return intp::InterpolationFunction1D<ORDER_OUT_, value_type,
                                             coord_type>(
            std::make_pair(psi_sample.front(), psi_sample.back()),
            intp::util::get_range(data), false);
    }
};

#endif  // SPDATA_H
