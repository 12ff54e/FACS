#ifndef FLOQUET_H
#define FLOQUET_H

#include <numbers>
#include "integrator.h"

// TODO: Use a more general form
template <typename T>
struct FloquetMatrix {
    using value_type = T;

    template <typename F>
    FloquetMatrix(const F& potential, value_type period, std::size_t steps) {
        auto even_f1 = integrate_linear_Homogeneous_2(potential, 0., period, 1.,
                                                      0., steps);
        auto odd_f1 = integrate_linear_Homogeneous_2(potential, 0., period, 0.,
                                                     1., steps);

        std::tie(a11, a12) = even_f1;
        std::tie(a21, a22) = odd_f1;
    }

    // For now, two eigenvalues are reciprocal to each other
    auto eigenvalue() const {
        auto del = std::sqrt(
            std::complex<value_type>{std::pow(a11 - a22, 2) + 4 * a12 * a21});
        auto tr = a11 + a22;
        return .5 * (tr - del);
    }

   private:
    value_type a11, a12, a21, a22;
};

#endif  // FLOQUET_H
