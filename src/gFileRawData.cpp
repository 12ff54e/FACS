#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "gFileRawData.h"
#include "util.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/console.h>
#endif

bool GFileRawData::is_complete() const noexcept {
    return complete_;
}

std::istream& operator>>(std::istream& is, GFileRawData& g) {
    std::string s_dum;
    std::string line;
    std::stringstream ss_line;
    double d_dum;
    // 1st line, extarcted with a fixed width style to avoid peculiar layout
    // causing problems
    std::getline(is, line);
    ss_line.str(line);
    ss_line.read(g.metadata.data(), 48);
    char str_tmp[5]{};
    ss_line.read(str_tmp, 4);
    ss_line.read(str_tmp, 4);
    g.nw = static_cast<unsigned>(atoi(str_tmp));
    ss_line.read(str_tmp, 4);
    g.nh = static_cast<unsigned>(atoi(str_tmp));
    if (!ss_line.eof()) {
        // There is extra metadata
        g.extra_metadata =
            std::string(std::istreambuf_iterator<char>(ss_line), {});
    }

    auto print = [](auto msg) {
#ifdef __EMSCRIPTEN__
        emscripten_out(msg);
#else
        std::cout << msg;
#endif
    };

    if (ss_line.fail()) {
        print("Data corruption at 1st line.\n");
        return is;
    }

    // 2nd line
    std::getline(is, line);
    ss_line.clear();
    ss_line.str(line);
    ss_line >> g.dim.x() >> g.dim.y() >> g.r_center >> g.r_left >> g.z_mid;
    if (ss_line.fail()) {
        print("Data corruption at 2nd line.\n");
        return is;
    }
    // 3rd line
    std::getline(is, line);
    ss_line.clear();
    ss_line.str(line);
    ss_line >> g.magnetic_axis.x() >> g.magnetic_axis.y() >>
        g.flux_magnetic_axis >> g.flux_LCFS >> g.b_center;
    if (ss_line.fail()) {
        print("Data corruption at 3rd line.\n");
        return is;
    }
    // 4th line
    std::getline(is, line);
    ss_line.clear();
    ss_line.str(line);
    ss_line >> g.current >> d_dum >> d_dum >> d_dum >> d_dum;
    if (ss_line.fail()) {
        print("Data corruption at 4th line.\n");
        return is;
    }
    // 5th line
    std::getline(is, line);
    ss_line.clear();
    ss_line.str(line);
    ss_line >> d_dum >> d_dum >> g.flux_sep >> g.sep.x() >> g.sep.y();
    if (ss_line.fail()) {
        print("Data corruption at 5th line.\n");
        return is;
    }

    auto read_vec = [&is](unsigned int count, std::vector<double>& arr) {
        arr.reserve(count);
        for (unsigned i = 0; i < count; ++i) {
            double d;
            is >> d;
            arr.emplace_back(d);
        }
    };

    // poloidal current
    read_vec(g.nw, g.f_pol);
    if (is.fail()) {
        print("Data corruption at poloidal current.\n");
        return is;
    }
    // pressure
    read_vec(g.nw, g.pressure);
    if (is.fail()) {
        print("Data corruption at pressure.\n");
        return is;
    }
    // f*f^{\prime}
    read_vec(g.nw, g.f_f_prime);
    if (is.fail()) {
        print("Data corruption at f*f'.\n");
        return is;
    }
    // p^{\prime}
    read_vec(g.nw, g.p_prime);
    if (is.fail()) {
        print("Data corruption at p'.\n");
        return is;
    }
    // flux
    g.flux.resize({g.nw, g.nh});
    for (unsigned j = 0; j < g.nh; ++j) {
        for (unsigned i = 0; i < g.nw; ++i) { is >> g.flux(i, j); }
    }
    if (is.fail()) {
        print("Data corruption at \\psi(r,z).\n");
        return is;
    }
    // safety factor
    read_vec(g.nw, g.safety_factor);
    if (is.fail()) {
        print("Data corruption at q.\n");
        return is;
    }
    // # boundary point and # limiter point
    is >> g.boundary_num >> g.limiter_num;
    if (is.fail()) {
        print("Data corruption at boundary number or limiter number.\n");
        return is;
    }
    // boundary points
    for (unsigned i = 0; i < g.boundary_num; ++i) {
        double rr, zz;
        is >> rr >> zz;
        g.boundary.emplace_back(rr, zz);
    }
    if (is.fail()) {
        print("Data corruption at boundary points.\n");
        return is;
    }

    g.rearrange_boundary();

    // limiter points
    for (unsigned i = 0; i < g.limiter_num; ++i) {
        double rr, zz;
        is >> rr >> zz;
        g.limiter.emplace_back(rr, zz);
    }
    if (is.fail()) {
        print("Data corruption at limiter points.\n");
        return is;
    }

    g.complete_ = true;

    return is;
}

void GFileRawData::rearrange_boundary() {
    geometric_poloidal_angles.reserve(boundary_num);

    size_t middle = 0;
    for (size_t i = 0; i < boundary_num; ++i) {
        geometric_poloidal_angles.push_back(
            util::arctan(boundary[i] - magnetic_axis));

        if (i > 0 && std::abs(geometric_poloidal_angles[i] -
                              geometric_poloidal_angles[i - 1]) > M_PI) {
            // middle is the index of the first angle crossing \theta=0
            middle = i;
        }
    }

    std::rotate(boundary.begin(),
                boundary.begin() + static_cast<std::ptrdiff_t>(middle),
                boundary.end());
    std::rotate(
        geometric_poloidal_angles.begin(),
        geometric_poloidal_angles.begin() + static_cast<std::ptrdiff_t>(middle),
        geometric_poloidal_angles.end());

    {
        auto last = std::unique(geometric_poloidal_angles.begin(),
                                geometric_poloidal_angles.end());
        geometric_poloidal_angles.erase(last, geometric_poloidal_angles.end());
    }
    {
        auto last = std::unique(boundary.begin(), boundary.end());
        boundary.erase(last, boundary.end());
    }

    // make sure boundary points are sorted counterclockwise
    if (geometric_poloidal_angles[0] > geometric_poloidal_angles[1]) {
        std::reverse(boundary.begin(), boundary.end());
        std::reverse(geometric_poloidal_angles.begin(),
                     geometric_poloidal_angles.end());
    }
}
