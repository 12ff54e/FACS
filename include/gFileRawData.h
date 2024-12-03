#ifndef GFILE_RAW_DATA_H
#define GFILE_RAW_DATA_H

#include <fstream>
#include <string>
#include <vector>

#include "intp.h"
#include "vec.h"

struct GFileRawData {
    std::array<char, 48> metadata;
    std::string extra_metadata;
    std::string code_name;
    std::string date;
    std::string shot_id;
    std::string time;

    // horizontal sample number
    unsigned int nw;
    // vertical sample number
    unsigned int nh;
    // horizontal and vertical dimension
    Vec<2, double> dim;
    // center position of horizontal range
    double r_center;
    // left position of horizontal range
    double r_left;
    // center position of vertical range
    double z_mid;
    // magnetic axis coordinate
    Vec<2, double> magnetic_axis;
    // magnetic flux at magnetic axis
    double flux_magnetic_axis;
    // magnetic flux at Last Closed Flux Surface
    double flux_LCFS;
    // magnetic field strength at center position
    double b_center;
    // current
    double current;
    // magnetic flux at separatrix
    double flux_sep;
    // separatrix coordinate
    Vec<2, double> sep;

    // poloidal current density profile
    std::vector<double> f_pol;
    // pressure profile
    std::vector<double> pressure;
    // f*f^{\prime}
    std::vector<double> f_f_prime;
    // p^{\prime}
    std::vector<double> p_prime;

    // magnetic flux
    intp::Mesh<double, 2> flux;

    // safety factor profile
    std::vector<double> safety_factor;

    // boundary point number
    unsigned int boundary_num;
    // limiter pointer number
    unsigned int limiter_num;

    // boundary point coordinates
    std::vector<Vec<2, double>> boundary;
    // limiter point coordinates
    std::vector<Vec<2, double>> limiter;

    // extra data

    std::vector<double> geometric_poloidal_angles;

    // constructors

    GFileRawData() : flux{1, 1}, complete_(false) {}
    GFileRawData(const GFileRawData&) = delete;
    GFileRawData(GFileRawData&&) = default;

    // properties

    bool is_complete() const noexcept;

    // post-process

    void rearrange_boundary();

    // operators

    friend std::istream& operator>>(std::istream&, GFileRawData&);

   private:
    // the completeness of raw data
    bool complete_;
};

#endif  // GFILE_RAW_DATA_H
