#ifndef CONTOUR_H
#define CONTOUR_H

#include <algorithm>
#include <vector>

#include "gFileRawData.h"
#include "util.h"
#include "vec.h"

template <typename T>
class Contour {
   public:
    using value_type = T;
    using pt_type = Vec<2, value_type>;

   private:
    const value_type flux_;
    std::vector<pt_type> pts_;

   public:
    template <typename Intp>
    Contour(value_type psi,
            const Intp& flux,
            const GFileRawData& gfile_raw_data)
        : flux_(psi) {
        pts_.reserve(gfile_raw_data.boundary.size());
        for (size_t i = 0; i < gfile_raw_data.boundary.size(); ++i) {
            pts_.emplace_back(
                util::vec_field_find_root(flux, gfile_raw_data.magnetic_axis,
                                          gfile_raw_data.boundary[i], psi));
        }
    }

    // properties

    size_t size() const noexcept { return pts_.size(); };

    double flux() const noexcept { return flux_; }

    // element access

    const Vec<2, double>& operator[](std::size_t i) const { return pts_[i]; }
};

#endif  // CONTOUR_H
