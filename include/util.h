#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "vec.h"

namespace util {

/**
 * @brief Sign function
 *
 * @return sign of val
 */
template <typename Tf>
inline int sgn(const Tf& val) {
    return (Tf(0) < val) - (val < Tf(0));
}

template <typename T>
inline T abs(T x) {
    return std::abs(x);
}

template <typename T>
inline T arctan(T x, T y) {
    T atan = std::atan2(y, x);
    return atan < 0 ? static_cast<T>(2 * static_cast<T>(M_PIl) + atan) : atan;
}

template <typename T>
inline T arctan(const Vec<2, T>& pt) {
    return arctan(pt.x(), pt.y());
}

template <typename T>
inline T lerp(T left, T right, T c) {
    return (static_cast<T>(1) - c) * left + c * right;
}

// auxilary functions for find root
namespace detail {

template <typename Func, typename Tx, typename Tf>
void bracket(Func& func, Tx& a, Tx& b, Tx c, Tf& fa, Tf& fb, Tf& d, Tf& fd) {
    Tf tol = std::numeric_limits<Tf>::epsilon() * 2;

    if ((b - a) < 2 * tol * a) {
        c = a + (b - a) / 2;
    } else if (c - a <= util::abs(a) * tol) {
        c = a + util::abs(a) * tol;
    } else if (b - c <= util::abs(b) * tol) {
        c = b - util::abs(b) * tol;
    }

    Tf fc = func(c);
    if (std::fpclassify(fc) == FP_ZERO) {
        a = c;
        fa = 0;
        d = 0;
        fd = 0;
        return;
    }

    if (sgn(fa) * sgn(fc) < 0) {
        d = b;
        fd = fb;
        b = c;
        fb = fc;
    } else {
        d = a;
        fd = fa;
        a = c;
        fa = fc;
    }
}

template <typename T>
T safe_div(T num, T denom, T r) {
    return util::abs(denom) < 1 &&
                   util::abs(denom * std::numeric_limits<T>::max()) <=
                       util::abs(num)
               ? r
               : num / denom;
}

template <typename Tx, typename Tf>
Tx secant_interpolate(const Tx& a, const Tx& b, const Tf& fa, const Tf& fb) {
    Tf tol = std::numeric_limits<Tf>::epsilon() * 5;
    Tx c = a - (fa / (fb - fa)) * (b - a);

    return c - a <= util::abs(a) * tol || b - c <= util::abs(b) * tol
               ? (a + b) / 2
               : c;
}

template <typename T>
bool outside(const T& c, const T& a, const T& b) {
    return c <= a || c >= b;
}

template <typename Tx, typename Tf>
Tx quadratic_interpolation(const Tx& a,
                           const Tx& b,
                           const Tx& d,
                           const Tf& fa,
                           const Tf& fb,
                           const Tf& fd,
                           unsigned count) {
    Tf B = safe_div(Tf(fb - fa), Tf(b - a), std::numeric_limits<Tf>::max());
    Tf A = safe_div(Tf(fd - fb), Tf(d - b), std::numeric_limits<Tf>::max());
    A = safe_div(Tf(A - B), Tf(d - a), Tf(0));

    if (std::fpclassify(A) == FP_ZERO) {
        return secant_interpolate(a, b, fa, fb);
    }

    Tx c = sgn(A) * sgn(fa) > 0 ? a : b;
    for (unsigned i = 1; i <= count; ++i) {
        c -= safe_div(Tf(fa + (B + A * (c - b)) * (c - a)),
                      Tf(B + A * (2 * c - a - b)), Tf(1 + c - a));
    }

    return outside(c, a, b) ? secant_interpolate(a, b, fa, fb) : c;
}

template <typename Tx, typename Tf>
Tx cubic_interpolation(const Tx& a,
                       const Tx& b,
                       const Tx& d,
                       const Tx& e,
                       const Tf& fa,
                       const Tf& fb,
                       const Tf& fd,
                       const Tf& fe) {
    Tx q11 = (d - e) * fd / (fe - fd);
    Tx q21 = (b - d) * fb / (fd - fb);
    Tx q31 = (a - b) * fa / (fb - fa);
    Tx d21 = (b - d) * fd / (fd - fb);
    Tx d31 = (a - b) * fb / (fb - fa);

    Tx q22 = (d21 - q11) * fb / (fe - fb);
    Tx q32 = (d31 - q21) * fa / (fd - fa);
    Tx d32 = (d31 - q21) * fd / (fd - fa);
    Tx q33 = (d32 - q22) * fa / (fe - fa);

    Tx c = q31 + q32 + q33 + a;

    return outside(c, a, b) ? quadratic_interpolation(a, b, d, fa, fb, fd, 3)
                            : c;
}
}  // namespace detail

/**
 * @brief Find root of a function using TOMS748. A simplified version adapted
 * from Boost library, ignoring most sanity checks.
 *
 * @tparam Tx input value type
 * @param ax left staring point
 * @param bx right starting point
 * @param tol a function accepting two value and determining them being within
 * tolerance
 */
template <typename Func, typename TolFunc, typename Tx>
Tx find_root(const Func& func, const Tx& ax, const Tx& bx, const TolFunc& tol) {
    using Tf = typename std::result_of<Func(Tx)>::type;
    // control parameters

    const unsigned max_iter = 50;
    unsigned count = max_iter;
    static const Tf mu{.5l};

    if (ax >= bx) { throw std::domain_error("Given interval do not exist."); }

    Tx a = ax, b = bx;
    Tf fa = func(a), fb = func(b);

    if (sgn(fa) * sgn(fb) > 0) {
        throw std::domain_error("Given interval may not bracket a root.");
    }

    if (tol(a, b) || (std::fpclassify(fa) == FP_ZERO) ||
        (std::fpclassify(fb) == FP_ZERO)) {
        return std::fpclassify(fb) == FP_ZERO ? b : a;
    }

    Tx d = 0;
    Tx e{1.e5l};
    Tf fe{1.e5l};
    Tf fd{1.e5l};

    Tx c = detail::secant_interpolate(a, b, fa, fb);
    detail::bracket(func, a, b, c, fa, fb, d, fd);
    --count;
    if (count != 0 && (std::fpclassify(fa) != FP_ZERO) && !(tol(a, b))) {
        c = detail::quadratic_interpolation(a, b, d, fa, fb, fd, 2);
        e = d;
        fe = fd;
        detail::bracket(func, a, b, c, fa, fb, d, fd);
        --count;
    }

    Tf min_diff = std::numeric_limits<Tf>::min() * 32;
    Tx u;
    Tf fu;
    while (count != 0 && (std::fpclassify(fa) != FP_ZERO) && !tol(a, b)) {
        Tx a0 = a;
        Tx b0 = b;

        bool prof = (util::abs(fa - fb) < min_diff) ||
                    (util::abs(fa - fd) < min_diff) ||
                    (util::abs(fa - fe) < min_diff) ||
                    (util::abs(fb - fd) < min_diff) ||
                    (util::abs(fb - fe) < min_diff) ||
                    (util::abs(fd - fe) < min_diff);
        c = prof ? detail::quadratic_interpolation(a, b, d, fa, fb, fd, 2)
                 : detail::cubic_interpolation(a, b, d, e, fa, fb, fd, fe);

        e = d;
        fe = fd;
        detail::bracket(func, a, b, c, fa, fb, d, fd);
        if ((0 == --count) || (std::fpclassify(fa) == FP_ZERO) || tol(a, b)) {
            break;
        }

        prof = (util::abs(fa - fb) < min_diff) ||
               (util::abs(fa - fd) < min_diff) ||
               (util::abs(fa - fe) < min_diff) ||
               (util::abs(fb - fd) < min_diff) ||
               (util::abs(fb - fe) < min_diff) ||
               (util::abs(fd - fe) < min_diff);
        c = prof ? detail::quadratic_interpolation(a, b, d, fa, fb, fd, 3)
                 : detail::cubic_interpolation(a, b, d, e, fa, fb, fd, fe);

        detail::bracket(func, a, b, c, fa, fb, d, fd);
        if ((0 == --count) || (std::fpclassify(fa) == FP_ZERO) || tol(a, b)) {
            break;
        }

        if (util::abs(fa) < util::abs(fb)) {
            u = a;
            fu = fa;
        } else {
            u = b;
            fu = fb;
        }
        c = u - 2 * (fu / (fb - fa)) * (b - a);
        if (util::abs(c - u) > (b - a) / 2) { c = a + (b - a) / 2; }

        e = d;
        fe = fd;
        detail::bracket(func, a, b, c, fa, fb, d, fd);
        if ((0 == --count) || (std::fpclassify(fa) == FP_ZERO) || tol(a, b)) {
            break;
        }

        if ((b - a) < mu * (b0 - a0)) { continue; }

        e = d;
        fe = fd;
        detail::bracket(func, a, b, Tx(a + (b - a) / 2), fa, fb, d, fd);
        --count;
    }

#ifdef _TRACE
    std::cout << "[TRACE] Iterate " << max_iter - count << " times.\n";
#endif
    return std::fpclassify(fb) == FP_ZERO ? b : a;
}

/**
 * @brief Find root of a function using TOMS748, with default tolerance function
 *
 */
template <typename Func, typename Tx>
Tx find_root(const Func& func, const Tx& ax, const Tx& bx) {
    return find_root(func, ax, bx, [](Tx a, Tx b) {
        return std::abs(a - b) < Tx{4} * std::numeric_limits<Tx>::epsilon() *
                                     std::min(std::abs(a), std::abs(b));
    });
}

/**
 * @brief Find root of $func(v) = field_val$ over vector by searching segment
 * between two given points
 *
 */
template <typename Func, std::size_t D, typename T>
Vec<D, T> vec_field_find_root(const Func& func,
                              const Vec<D, T>& v1,
                              const Vec<D, T>& v2,
                              T field_val = T{}) {
    T w = find_root(
        [&](T w_) { return func((T{1} - w_) * v1 + w_ * v2) - field_val; }, T{},
        T{1});
    return (T{1} - w) * v1 + w * v2;
}

// auxiliary function for Gauss-Kronrod quadrature
namespace detail {

/**
 * @brief The storage for abscissa and weight of order N Gauss-Kronrod
 * integration method. The data for N=5, 15 is precomputed.
 *
 */
template <size_t N>
struct gauss_kronrod_detail {
    constexpr static std::array<double, (N + 1) / 2> abscissa();
    constexpr static std::array<double, ((N - 1) / 2 + 1) / 2> gauss_weight();
    constexpr static std::array<double, (N + 1) / 2> kronrod_weight();
};

template <>
struct gauss_kronrod_detail<5> {
    constexpr static std::array<double, 3> abscissa() {
        return {0., 0.57735026918962576, 0.92582009977255146};
    }

    constexpr static std::array<double, 1> gauss_weight() { return {1.}; }

    constexpr static std::array<double, 3> kronrod_weight() {
        return {
            0.62222222222222222,
            0.49090909090909091,
            0.19797979797979798,
        };
    }
};

template <>
struct gauss_kronrod_detail<15> {
    constexpr static std::array<double, 8> abscissa() {
        return {
            0.,
            0.20778495500789847,
            0.40584515137739717,
            0.58608723546769113,
            0.74153118559939444,
            0.86486442335976907,
            0.94910791234275852,
            0.99145537112081264,
        };
    }
    constexpr static std::array<double, 4> gauss_weight() {
        return {
            0.41795918367346939,
            0.38183005050511894,
            0.27970539148927667,
            0.12948496616886969,
        };
    }
    constexpr static std::array<double, 8> kronrod_weight() {
        return {
            2.09482141084727828e-01, 2.04432940075298892e-01,
            1.90350578064785410e-01, 1.69004726639267903e-01,
            1.40653259715525919e-01, 1.04790010322250184e-01,
            6.30920926299785533e-02, 2.29353220105292250e-02,
        };
    }
};

/**
 * @brief Order N Gauss-Kronrod quadrature, with embedded Gauss quadrature order
 * = (N-1)/2
 *
 */
template <size_t N, typename Tx>
struct gauss_kronrod : gauss_kronrod_detail<N> {
    using base = gauss_kronrod_detail<N>;

    /**
     * @brief Core function of gauss-kronrod integrate method, it integrates the
     * given function on [-1, 1].
     *
     * @tparam Func Function type
     * @param func Integrand
     * @return a pair of integral and err
     */
    template <typename Func>
    auto static gauss_kronrod_basic(const Func& func)
        -> std::pair<decltype(std::declval<Func>()(std::declval<Tx>())), Tx> {
        auto f0 = func(Tx{});
        using Ty = decltype(f0);
        constexpr auto gauss_order = (N - 1) / 2;

        Ty gauss_integral =
            gauss_order & 1 ? base::gauss_weight()[0] * f0 : Ty{};
        Ty kronrod_integral = base::kronrod_weight()[0] * f0;

        for (size_t i = 1; i < base::abscissa().size(); ++i) {
            Ty f = func(base::abscissa()[i]) + func(-base::abscissa()[i]);
            gauss_integral +=
                (gauss_order - i) & 1 ? base::gauss_weight()[i / 2] * f : Ty{};
            kronrod_integral += base::kronrod_weight()[i] * f;
        }

        return std::make_pair(
            kronrod_integral,
            std::max(
                static_cast<Tx>(std::abs(kronrod_integral - gauss_integral)),
                static_cast<Tx>(std::abs(kronrod_integral) *
                                std::numeric_limits<Tx>::epsilon() * 2)));
    }

    template <typename Func>
    auto static gauss_kronrod_adaptive(const Func& func,
                                       Tx a,
                                       Tx b,
                                       size_t max_subdivide,
                                       Tx abs_tol,
                                       Tx global_rel_tol)
        -> decltype(std::declval<Func>()(std::declval<Tx>())) {
        // call stack
        std::vector<std::array<Tx, 2>> pending_intervals;
        // quadrature sum
        decltype(std::declval<Func>()(std::declval<Tx>())) sum{};

        Tx inv_scale = 2. / (b - a);
        pending_intervals.push_back({a, b});
        while (!pending_intervals.empty()) {
            auto [l, r] = pending_intervals.back();
            pending_intervals.pop_back();

            const Tx mid = (r + l) / 2;
            const Tx scale = (r - l) / 2;
            auto normalize_func = [&](Tx x) { return func(scale * x + mid); };
            auto result = gauss_kronrod_basic(normalize_func);
            auto integral = result.first * scale;
            auto err = result.second * scale;

            if (std::fpclassify(abs_tol) == FP_ZERO) {
                abs_tol = std::abs(global_rel_tol * integral);
            }
            if (std::ldexp(scale, static_cast<int>(max_subdivide)) >
                    0.99 * (b - a) &&
                err > abs_tol * inv_scale &&
                err > std::abs(global_rel_tol * integral)) {
                pending_intervals.push_back({mid, r});
                pending_intervals.push_back({l, mid});
            } else {
                sum += integral;
            }
        }

        return sum;
    }
};

}  // namespace detail

template <typename Func, typename Tx>
auto integrate(const Func& func,
               Tx a,
               Tx b,
               Tx tol = std::sqrt(std::numeric_limits<Tx>::epsilon()),
               size_t max_subdivide = 15)
    -> decltype(std::declval<Func>()(std::declval<Tx>())) {
    using impl = detail::gauss_kronrod<15, Tx>;

    return impl::gauss_kronrod_adaptive(func, a, b, max_subdivide, Tx{}, tol);
}

template <typename Func, typename Tx>
auto integrate_coarse(const Func& func,
                      Tx a,
                      Tx b,
                      Tx tol = std::sqrt(std::numeric_limits<Tx>::epsilon()),
                      size_t max_subdivide = 3)
    -> decltype(std::declval<Func>()(std::declval<Tx>())) {
    using impl = detail::gauss_kronrod<5, Tx>;

    return impl::gauss_kronrod_adaptive(func, a, b, max_subdivide, Tx{}, tol);
}

}  // namespace util

#endif  // UTIL_H
