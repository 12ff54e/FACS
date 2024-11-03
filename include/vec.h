#ifndef MCT_VEC_
#define MCT_VEC_

#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

/**
 * @brief A vector in Euclidean space.
 *
 * @tparam D Dimension of the underlying space.
 * @tparam T Type of coordinate.
 */
template <std::size_t D, typename T>
class Vec;

template <std::size_t D, typename T>
class Vec_base {
    static_assert(D > 0, "D cannot be 0.");

   public:
    static constexpr std::size_t dim = D;
    using value_type = T;
    using vec_type = Vec<dim, value_type>;

   protected:
    std::array<value_type, dim> coord;

   public:
    constexpr Vec_base() = default;

    template <typename... Ts,
              typename = typename std::enable_if<sizeof...(Ts) == dim>::type>
    constexpr Vec_base(Ts... vs) noexcept
        : coord{static_cast<value_type>(vs)...} {}

    /**
     * @brief Conversion constructor from supported type
     *
     * @param other another vec
     */
    template <typename U,
              typename = typename std::enable_if<
                  std::is_convertible<U, T>::value>::type>
    Vec_base(Vec_base<dim, U> other) noexcept {
        for (std::size_t i = 0; i < dim; ++i) {
            coord[i] = static_cast<value_type>(other[i]);
        }
    }

    // zero element

    static constexpr vec_type zero() noexcept {
        return Vec_base<dim, value_type>{};
    }

    // element access

    T& operator[](std::size_t i) noexcept { return coord[i]; }

    const T& operator[](std::size_t i) const noexcept { return coord[i]; }

    // comparison operations

#if __cplusplus >= 202002L
    friend bool operator==(const vec_type& lhs, const vec_type& rhs) {
        auto equal_aux = []<std::size_t... indices>(
                             std::index_sequence<indices...>,
                             const vec_type& lhs_, const vec_type& rhs_) {
            if constexpr (std::is_arithmetic_v<typename vec_type::value_type>) {
                return (... && (std::fpclassify(lhs_[indices] -
                                                rhs_[indices]) == FP_ZERO));
            } else {
                return (... && (lhs_[indices] == rhs_[indices]));
            }
        };
        return equal_aux(std::make_index_sequence<dim>{}, lhs, rhs);
    }

    friend bool operator!=(const vec_type& lhs, const vec_type& rhs) {
        return !operator==(lhs, rhs);
    }
#else
    bool operator==(const vec_type& rhs) const noexcept {
        for (std::size_t i = 0; i < dim; ++i) {
            if (coord[i] != rhs[i]) { return false; }
        }
        return true;
    }
    bool operator!=(const vec_type& rhs) const noexcept {
        return !operator==(rhs);
    }
#endif
    // arithmetic operations
    friend vec_type& operator+=(vec_type& lhs, const vec_type& rhs) noexcept {
        for (std::size_t i = 0; i < dim; ++i) { lhs[i] += rhs[i]; }
        return lhs;
    }
    // vec_type& operator+=(const vec_type& rhs) noexcept {
    //     for (std::size_t i = 0; i < dim; ++i) { coord[i] += rhs[i]; }
    //     return *reinterpret_cast<vec_type*>(this);
    // }
    vec_type& operator-=(const vec_type& rhs) noexcept {
        for (std::size_t i = 0; i < dim; ++i) { coord[i] -= rhs[i]; }
        return *reinterpret_cast<vec_type*>(this);
    }

    vec_type& operator*=(const T& scalar) noexcept {
        for (std::size_t i = 0; i < dim; ++i) { coord[i] *= scalar; }
        return *reinterpret_cast<vec_type*>(this);
    }

    vec_type& operator/=(const T& scalar) noexcept {
        for (std::size_t i = 0; i < dim; ++i) { coord[i] /= scalar; }
        return *reinterpret_cast<vec_type*>(this);
    }

    friend vec_type operator+(vec_type lhs, const vec_type& rhs) noexcept {
        lhs += rhs;
        return lhs;
    }
    friend vec_type operator-(vec_type lhs, const vec_type& rhs) noexcept {
        lhs -= rhs;
        return lhs;
    }

    friend vec_type operator*(vec_type vec, const T& scalar) noexcept {
        return vec *= scalar;
    }
    friend vec_type operator*(const T& scalar, vec_type vec) noexcept {
        return vec *= scalar;
    }

    friend vec_type operator/(vec_type vec, const T& scalar) noexcept {
        return vec /= scalar;
    }

    // conversion operator

    // convert to the underlying coordinate array
    constexpr operator std::array<value_type, dim>() const { return coord; }

    // convert to derived class
    constexpr operator vec_type() { return static_cast<vec_type&&>(*this); }

    // properties

    T L2_norm_square_() const {
        T norm{};
        for (auto& c : coord) { norm += c * c; }
        return norm;
    }

    T mag() const { return std::sqrt(L2_norm_square_()); }
};

/**
 * @brief Generic Vector type of any dimension, default to consisting double as
 * coordinates
 *
 * @tparam D Dimension
 * @tparam T Underlying types of each dimension
 */
template <std::size_t D, typename T = double>
class Vec : public Vec_base<D, T> {};

template <typename T>
class Vec<2, T> : public Vec_base<2, T> {
   public:
    using Vec_base<2, T>::Vec_base;

    T& x() noexcept { return this->coord[0]; }
    T& y() noexcept { return this->coord[1]; }

    const T& x() const noexcept { return this->coord[0]; }
    const T& y() const noexcept { return this->coord[1]; }
};

template <typename T>
class Vec<3, T> : public Vec_base<3, T> {
   public:
    using Vec_base<3, T>::Vec_base;

    T& x() noexcept { return this->coord[0]; }
    T& y() noexcept { return this->coord[1]; }
    T& z() noexcept { return this->coord[2]; }

    const T& x() const noexcept { return this->coord[0]; }
    const T& y() const noexcept { return this->coord[1]; }
    const T& z() const noexcept { return this->coord[2]; }
};

template <typename T = double>
Vec<2, T> cross(const Vec<2, T>& vec) {
    return Vec<2, T>{-vec.y(), vec.x()};
}

template <typename T = double>
Vec<3, T> cross(const Vec<3, T>& v1, const Vec<3, T>& v2) {
    return Vec<3, T>{v1.y() * v2.z() - v1.z() * v2.y(),
                     v1.z() * v2.x() - v1.x() * v2.z(),
                     v1.x() * v2.y() - v1.y() * v2.x()};
}

namespace vec_impl {

#if __cplusplus < 202002L
template <typename T, std::size_t... indices>
T dot_aux(std::index_sequence<indices...>,
          const Vec<sizeof...(indices), T>& v1,
          const Vec<sizeof...(indices), T>& v2) {
    return (... + (v1[indices] * v2[indices]));
}
#endif

}  // namespace vec_impl

template <std::size_t D, typename T = double>
T dot(const Vec<D, T>& v1, const Vec<D, T>& v2) {
#if __cplusplus >= 202002L
    auto dot_aux = []<std::size_t... indices>(
                       std::index_sequence<indices...>,
                       const Vec<sizeof...(indices), T>& v1_,
                       const Vec<sizeof...(indices), T>& v2_) {
        return (... + (v1_[indices] * v2_[indices]));
    };
#endif
    using namespace vec_impl;
    return dot_aux(std::make_index_sequence<D>{}, v1, v2);
}

#endif
