#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <array>
#include <cmath>
#include <complex>
#include <vector>

template <typename T>
struct Integrator {
    using state_type = T;
    using value_type = typename state_type::value_type;
    using velocity_type = typename state_type::velocity_type;
    static constexpr std::size_t order = 3;

    Integrator(state_type& initial_state,
               value_type dt = 1.,
               value_type upper_err_bound = 1.e-7,
               value_type lower_err_bound = 1.e-10)
        : current_dt(dt),
          state(initial_state),
          upper_err_bound_(upper_err_bound),
          lower_err_bound_(lower_err_bound),
          intermediates(([&]<auto... p_idx>(std::index_sequence<p_idx...>) {
              return std::array<velocity_type, order>{
                  ((void)p_idx, state.initial_velocity_storage())...};
          })(std::make_index_sequence<order>{})) {}

    void step(value_type dt) {
        ([&]<auto... p_idx>(std::index_sequence<p_idx...>) {
            (([&]<auto... k_idx>(std::index_sequence<k_idx...>) {
                 constexpr auto p = p_idx;
                 state.put_velocity(intermediates[p]);
                 state.update((... + (coef[p][k_idx] * intermediates[k_idx])),
                              coef[p][p + 1] * dt);
             })(std::make_index_sequence<p_idx + 1>{}),
             ...);
        })(std::make_index_sequence<order>{});
    }

    auto step_adaptive(value_type maximum_dt) {
        constexpr value_type beta = 0.9;
        auto state_current = state;

        value_type err{};
        while (true) {
            step(current_dt);
            err = ([&]<auto... k_idx>(std::index_sequence<k_idx...>) {
                return state.get_update_err(
                    (... + (coef[order][k_idx] * intermediates[k_idx])),
                    current_dt);
            })(std::make_index_sequence<3>{});
            if (err < upper_err_bound_) {
                current_dt *= beta * std::sqrt(upper_err_bound_ / err);
                break;
            }

            current_dt = std::min(
                maximum_dt,
                current_dt * beta * std::pow(upper_err_bound_ / err, 1. / 3));

            state = state_current;
        }

        return current_dt;
    }

   private:
    value_type current_dt;
    state_type& state;
    value_type upper_err_bound_;
    value_type lower_err_bound_;
    std::array<velocity_type, 3> intermediates;

    static inline constexpr std::array<std::array<double, 4>, 4> coef{
        {{1, 0.62653829327080},
         {0, 1, -0.55111240553326},
         {0, 1.5220585509963, -0.52205855099628, 0.92457411226246},
         {1., 0.13686116839369, -1.1368611683937}}};
};

template <typename T, typename F>
auto integrateLinearHomogeneous2(const F& coef,
                                 T t0,
                                 T t1,
                                 T f0,
                                 T f0_prime,
                                 std::size_t steps) {
    struct State {
        using value_type = T;
        // exploite complex as a 2D linear space
        using velocity_type = std::complex<T>;

        // d^2f/dt^2 = -coef*f
        void put_velocity(velocity_type& v) {
            v.real(f.imag());
            v.imag(-potential(t) * f.real());
        }

        void update(velocity_type v, value_type dt) {
            f += v * dt;
            t += dt;
        }

        auto get_update_err(velocity_type v, value_type dt) {
            auto df = v * dt;
            return std::max(std::abs(df.real()), std::abs(df.imag()));
        }

        auto initial_velocity_storage() { return velocity_type{}; }

        T t;
        velocity_type f;
        const F& potential;

        decltype(auto) operator=(const State& other) {
            t = other.t;
            f = other.f;
            return *this;
        }
    } state{t0, std::complex<T>{f0, f0_prime}, coef};

    T dt = (t1 - t0) / steps;
    Integrator integrator(state, dt);
    for (std::size_t i = 0; i < steps; ++i) { integrator.step(dt); }

    // adaptive integration, might not necessary
    // while (true) {
    //     t0 += integrator.step_adaptive(t1 - t0);

    //     if (t0 >= t1) { break; }
    // }

    return state.f.real();
}

#endif  // INTEGRATOR_H
