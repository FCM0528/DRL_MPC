#pragma once
#include <filesystem>
#include <limits>
#include <vector>

struct ReferencePoint {
    // CarSim's actual reference-table domain; used by all plot views.
    double x_min = std::numeric_limits<double>::quiet_NaN();
    double x_max = std::numeric_limits<double>::quiet_NaN();
    double y = 0.0;             // y_ref [m]
    double dydx = 0.0;          // first derivative
    double d2ydx2 = 0.0;        // second derivative
    double psi = 0.0;           // atan(dydx) [rad]
    double curvature = 0.0;     // [1/m]
};

class DoubleLaneChange {
public:
    // This Run_all.par mode is for a straight road aligned with global +X.
    // The CarSim LTARG_TABLE station/lateral coordinates then equal X/Y.
    static DoubleLaneChange fromCarSimParsfile(const std::filesystem::path& filename);
    ReferencePoint at(double x) const;
    double controlCurvature(double x, double sigma_m) const;

private:
    struct PathSample { double x, y, d2ydx2; };
    DoubleLaneChange() = default;
    std::vector<PathSample> path_;
};
