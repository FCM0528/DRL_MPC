#include "reference.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

DoubleLaneChange DoubleLaneChange::fromCarSimParsfile(
    const std::filesystem::path& filename) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot open CarSim reference Parsfile: " + filename.string());
    DoubleLaneChange result;
    bool in_table = false, table_found = false, straight_east = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.find("set_description path_id Straight East") != std::string::npos)
            straight_east = true;
        std::istringstream fields(line);
        std::string keyword;
        fields >> keyword;
        if (keyword == "LTARG_TABLE") {
            if (table_found) throw std::runtime_error("Multiple CarSim LTARG_TABLEs are not supported");
            std::string interpolation;
            fields >> interpolation;
            if (interpolation != "SPLINE_FLAT")
                throw std::runtime_error("CarSim reference requires LTARG_TABLE SPLINE_FLAT");
            table_found = in_table = true;
            continue;
        }
        if (!in_table) continue;
        if (keyword == "ENDTABLE") { in_table = false; continue; }
        if (keyword.empty() || keyword[0] == '!' || keyword[0] == '#') continue;
        std::istringstream values(line);
        double x = 0.0, y = 0.0;
        char comma = 0;
        if (!(values >> x >> comma >> y) || comma != ',' ||
            !std::isfinite(x) || !std::isfinite(y) ||
            (!result.path_.empty() && x <= result.path_.back().x))
            throw std::runtime_error("Invalid CarSim LTARG_TABLE row: " + line);
        result.path_.push_back({x, y, 0.0});
    }
    if (in_table || result.path_.size() < 3)
        throw std::runtime_error("CarSim reference LTARG_TABLE is missing or incomplete");
    if (!straight_east)
        throw std::runtime_error("CarSim LTARG_TABLE station is not global X on this road; "
                                 "a global X/Y path adapter is required");

    // Natural cubic spline through the path samples. Outside the CarSim table,
    // SPLINE_FLAT holds the endpoint lateral offset constant.
    const auto n = result.path_.size();
    std::vector<double> lower(n, 0.0), diagonal(n, 1.0), upper(n, 0.0), rhs(n, 0.0);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const double h0 = result.path_[i].x - result.path_[i - 1].x;
        const double h1 = result.path_[i + 1].x - result.path_[i].x;
        lower[i] = h0;
        diagonal[i] = 2.0 * (h0 + h1);
        upper[i] = h1;
        rhs[i] = 6.0 * ((result.path_[i + 1].y - result.path_[i].y) / h1 -
                        (result.path_[i].y - result.path_[i - 1].y) / h0);
    }
    for (std::size_t i = 1; i < n; ++i) {
        const double factor = lower[i] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    result.path_[n - 1].d2ydx2 = rhs[n - 1] / diagonal[n - 1];
    for (std::size_t i = n - 1; i-- > 0;)
        result.path_[i].d2ydx2 = (rhs[i] - upper[i] * result.path_[i + 1].d2ydx2) /
                                    diagonal[i];
    return result;
}

ReferencePoint DoubleLaneChange::at(double x) const {
    ReferencePoint ref;
    if (x <= path_.front().x || x >= path_.back().x) {
        ref.y = x <= path_.front().x ? path_.front().y : path_.back().y;
        return ref;
    }
    const auto right = std::upper_bound(path_.begin(), path_.end(), x,
        [](double value, const PathSample& p) { return value < p.x; });
    const auto& left = *(right - 1);
    const double h = right->x - left.x;
    const double a = (right->x - x) / h;
    const double b = (x - left.x) / h;
    ref.y = a * left.y + b * right->y +
            ((a*a*a - a) * left.d2ydx2 + (b*b*b - b) * right->d2ydx2) * h*h / 6.0;
    ref.dydx = (right->y - left.y) / h +
                (-(3*a*a - 1) * left.d2ydx2 +
                   (3*b*b - 1) * right->d2ydx2) * h / 6.0;
    ref.d2ydx2 = a * left.d2ydx2 + b * right->d2ydx2;
    ref.psi = std::atan(ref.dydx);
    ref.curvature = ref.d2ydx2 /
                    std::pow(1.0 + ref.dydx * ref.dydx, 1.5);
    return ref;
}
