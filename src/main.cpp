#include <iostream>
#include <numbers>
#include "Floquet.h"
#include "integrator.h"

int main() {
    auto mathieu_potential = [](double t) { return .5 - .4 * std::cos(2 * t); };
    constexpr std::size_t steps = 100;

    const FloquetMatrix flo_mat(mathieu_potential, std::numbers::pi, steps);
    auto eigenvalue = flo_mat.eigenvalue();

    std::cout << eigenvalue << '\n';

    return 0;
}
