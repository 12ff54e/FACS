#include <iostream>
#include <numbers>
#include "integrator.h"

int main() {
    auto mathieu_potential = [](double t) { return .5 - .2 * std::cos(2 * t); };
    constexpr std::size_t steps = 100;

    std::cout << integrateLinearHomogeneous2(mathieu_potential, 0.,
                                             std::numbers::pi, 1., 0., steps)
              << ", "
              << integrateLinearHomogeneous2(mathieu_potential, 0.,
                                             std::numbers::pi, 0., 1., steps)
              << '\n';

    return 0;
}
