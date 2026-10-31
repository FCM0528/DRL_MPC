#include "vehicle.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
constexpr double rad2deg = 180.0 / 3.14159265358979323846264338327950288;

std::vector<double> parseRow(const std::string& line, const std::string& file,
                             int line_number, bool header) {
    std::vector<double> values;
    std::size_t start = 0;
    while (start <= line.size()) {
        const auto end = line.find(',', start);
        const std::string field = line.substr(start, end - start);
        if (header && values.empty() && field == "slip_deg") {
            values.push_back(0.0);
        } else {
            std::size_t used = 0;
            double value = 0.0;
            try { value = std::stod(field, &used); }
            catch (const std::exception&) {
                throw std::runtime_error(file + ":" + std::to_string(line_number) +
                                         ": invalid tire table number");
            }
            if (used != field.size() || !std::isfinite(value))
                throw std::runtime_error(file + ":" + std::to_string(line_number) +
                                         ": invalid tire table number");
            values.push_back(value);
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return values;
}

std::size_t lowerIndex(const std::vector<double>& grid, double value) {
    const auto upper = std::upper_bound(grid.begin(), grid.end(), value);
    if (upper == grid.begin()) return 0;
    if (upper == grid.end()) return grid.size() - 2;
    return static_cast<std::size_t>(upper - grid.begin() - 1);
}

}

// Tire-table loading and interpolation share this translation unit with the
// vehicle plant so both use exactly the same force curve.
TireTable::TireTable(const std::filesystem::path& csv_file) {
    std::ifstream in(csv_file);
    if (!in) throw std::runtime_error("Cannot open tire table: " + csv_file.string());
    const std::string file = csv_file.string();
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Empty tire table: " + file);
    if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xef &&
        static_cast<unsigned char>(line[1]) == 0xbb &&
        static_cast<unsigned char>(line[2]) == 0xbf)
        line.erase(0, 3); // UTF-8 BOM written by some Windows spreadsheet editors.
    if (!line.empty() && line.back() == '\r') line.pop_back();
    auto header = parseRow(line, file, 1, true);
    if (header.size() < 3 || header.front() != 0.0)
        throw std::runtime_error("Tire table needs a zero/slip_deg corner and at least two loads");
    loads_.assign(header.begin() + 1, header.end());
    if (loads_.front() <= 0.0 || !std::is_sorted(loads_.begin(), loads_.end()) ||
        std::adjacent_find(loads_.begin(), loads_.end()) != loads_.end())
        throw std::runtime_error("Tire table loads must be positive and increasing");

    int line_number = 1;
    while (std::getline(in, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto values = parseRow(line, file, line_number, false);
        if (values.size() != loads_.size() + 1)
            throw std::runtime_error(file + ":" + std::to_string(line_number) +
                                     ": wrong number of force columns");
        const double slip = values.front();
        if (slip < 0.0 || (!slip_deg_.empty() && slip <= slip_deg_.back()))
            throw std::runtime_error("Tire table slip angles must increase from zero");
        if (std::any_of(values.begin() + 1, values.end(),
                        [](double force) { return force < 0.0; }))
            throw std::runtime_error("Tire table force magnitudes must be nonnegative");
        slip_deg_.push_back(slip);
        forces_.emplace_back(values.begin() + 1, values.end());
    }
    // Some source exports start at 0.5 deg. Complete the odd-symmetric tire
    // curve at zero without altering the user's raw CSV.
    if (!slip_deg_.empty() && slip_deg_.front() > 0.0) {
        slip_deg_.insert(slip_deg_.begin(), 0.0);
        forces_.insert(forces_.begin(), std::vector<double>(loads_.size(), 0.0));
    }
    // The alternate TireFy export appends a 26-deg row identical to 25 deg.
    // It is a hold marker, not a measured falling slope: retain 25 deg as the
    // last data point so the requested 25-to-35-deg extension keeps decreasing.
    if (slip_deg_.size() >= 3 && slip_deg_.back() == 26.0 &&
        slip_deg_[slip_deg_.size() - 2] == 25.0 &&
        forces_.back() == forces_[forces_.size() - 2]) {
        slip_deg_.pop_back();
        forces_.pop_back();
    }
    if (slip_deg_.size() < 2 || slip_deg_.front() != 0.0 ||
        std::any_of(forces_.front().begin(), forces_.front().end(),
                    [](double force) { return force != 0.0; }))
        throw std::runtime_error("Tire table needs a zero-force row at zero slip");
}

double TireTable::positiveForce(double slip_deg, double normal_load_n) const {
    if (!std::isfinite(normal_load_n) || normal_load_n < minLoad() ||
        normal_load_n > maxLoad())
        throw std::invalid_argument("Per-tire normal load is outside tire table range");
    // Continue the measured last-segment slope for at most 10 degrees (25 to
    // 35 degrees for the supplied table). Beyond that, hold the endpoint;
    // the extension is a model assumption, not another measured CSV row.
    constexpr double extension_deg = 10.0;
    const double angle = std::min(slip_deg, slip_deg_.back() + extension_deg);
    const auto i = lowerIndex(slip_deg_, angle);
    const auto j = lowerIndex(loads_, normal_load_n);
    const double ta = (angle - slip_deg_[i]) / (slip_deg_[i + 1] - slip_deg_[i]);
    const double tf = (normal_load_n - loads_[j]) / (loads_[j + 1] - loads_[j]);
    const double low = forces_[i][j] + tf * (forces_[i][j + 1] - forces_[i][j]);
    const double high = forces_[i + 1][j] + tf * (forces_[i + 1][j + 1] - forces_[i + 1][j]);
    return std::max(0.0, low + ta * (high - low));
}

double TireTable::corneringStiffness(double normal_load_n) const {
    const double first_slip_rad = slip_deg_[1] / rad2deg;
    return positiveForce(slip_deg_[1], normal_load_n) / first_slip_rad;
}

double TireTable::lateralForce(double slip_rad, double normal_load_n) const {
    if (!std::isfinite(slip_rad)) throw std::invalid_argument("Nonfinite tire slip");
    if (slip_rad == 0.0) return 0.0;
    return std::copysign(positiveForce(std::abs(slip_rad) * rad2deg, normal_load_n), slip_rad);
}

double TireTable::peakLateralForce(double normal_load_n) const {
    double peak = 0.0;
    for (const double slip : slip_deg_)
        peak = std::max(peak, positiveForce(slip, normal_load_n));
    return peak;
}

State6 matrixTimesState(const Matrix6& a, const State6& x) {
    State6 y{};
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) y[i] += a[i][j] * x[j];
    return y;
}

// 计算两个角度 a 和 b 的差值，结果在 [-π, π] 范围内
double angleDifference(double a, double b) {
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

// 车辆动力学模型的构造函数，检查参数合法性
std::pair<double, double> staticWheelLoads(const VehicleParameters& p) {
    constexpr double gravity = 9.80665;
    const double wheelbase = p.a + p.b;
    return {p.mass * gravity * p.b / (2.0 * wheelbase),
            p.mass * gravity * p.a / (2.0 * wheelbase)};
}

DiscreteErrorModel discretizeErrorModel(const VehicleParameters& p,
                                        const TireTable& tires,
                                        const VehicleState& op,
                                        double steer, double ax, double dt, bool nonlinear_tire,
                                        double tire_slope_floor) {
    // CarSim-coordinate three-DOF state: [vy,vx,psi,r,Y,X].
    // Cf/Cr are axle sums of the two per-wheel cornering forces.
    // The unobserved longitudinal tire/driver force is held at the value that
    // reproduces the measured CarSim body longitudinal acceleration at op.
    const double u = op.vx, v = op.vy, yaw = op.psi, r = op.r;
    if (!(std::isfinite(u) && u >= 1.0 && std::isfinite(v) &&
          std::isfinite(yaw) && std::isfinite(r) &&
          std::isfinite(steer) && std::isfinite(ax) &&
          std::isfinite(dt) && dt > 0.0))
        throw std::invalid_argument("Invalid three-DOF operating point");
    const auto [front_load, rear_load] = staticWheelLoads(p);
    const double alpha_f = steer - (v + p.a*r) / u;
    const double alpha_r = (p.b*r - v) / u;
    const auto slope = [&](double alpha, double load) {
        constexpr double h = 1e-5;
        const double local = (tires.lateralForce(alpha + h, load) - tires.lateralForce(alpha - h, load)) / (2*h);
        // Preserve control direction when a saturated or extrapolated table has
        // a zero/negative local slope. The force at the operating point remains exact.
        return std::max(local, tire_slope_floor * tires.corneringStiffness(load));
    };
    const double cf = 2.0 * p.road_grip_scale * (nonlinear_tire ? slope(alpha_f, front_load) : tires.corneringStiffness(front_load));
    const double cr = 2.0 * p.road_grip_scale * (nonlinear_tire ? slope(alpha_r, rear_load) : tires.corneringStiffness(rear_load));
    const double fyf = nonlinear_tire ? 2.0 * p.road_grip_scale * tires.lateralForce(alpha_f, front_load) : cf * alpha_f;
    const double fyr = nonlinear_tire ? 2.0 * p.road_grip_scale * tires.lateralForce(alpha_r, rear_load) : cr * alpha_r;
    const double fyf_u = cf * (v + p.a*r) / (u*u);
    const double fyf_v = -cf / u;
    const double fyf_r = -p.a*cf / u;
    const double fyr_u = cr * (v - p.b*r) / (u*u);
    const double fyr_v = -cr / u;
    const double fyr_r = p.b*cr / u;
    const double cpsi = std::cos(yaw), spsi = std::sin(yaw);

    Matrix6 jac{};
    State6 input{};
    // Longitudinal equation: m*u_dot=m*v*r+Fx_total-Fyf*delta_f.
    jac[1][0] = r - fyf_v*steer/p.mass;
    jac[1][1] = -fyf_u*steer/p.mass;
    jac[1][3] = v - fyf_r*steer/p.mass;
    input[1] = -(cf*steer + fyf)/p.mass;
    // Lateral and yaw equations use small-angle linearized tire forces.
    jac[0][0] = (fyf_v + fyr_v)/p.mass;
    jac[0][1] = -r + (fyf_u + fyr_u)/p.mass;
    jac[0][3] = -u + (fyf_r + fyr_r)/p.mass;
    input[0] = cf/p.mass;
    jac[2][3] = 1.0;
    jac[3][0] = (p.a*fyf_v - p.b*fyr_v)/p.iz;
    jac[3][1] = (p.a*fyf_u - p.b*fyr_u)/p.iz;
    jac[3][3] = (p.a*fyf_r - p.b*fyr_r)/p.iz;
    input[3] = p.a*cf/p.iz;
    // CarSim uses global X forward and Y left; preserve this rotation.
    jac[4][0] = cpsi;
    jac[4][1] = spsi;
    jac[4][2] = u*cpsi - v*spsi;
    jac[5][0] = -spsi;
    jac[5][1] = cpsi;
    jac[5][2] = -u*spsi - v*cpsi;

    const State6 x0{v,u,yaw,r,op.y,op.x};
    const State6 f0{(fyf+fyr)/p.mass-u*r, ax, r,
                    (p.a*fyf-p.b*fyr)/p.iz,
                    u*spsi+v*cpsi, u*cpsi-v*spsi};
    const auto linear_part = matrixTimesState(jac,x0);
    DiscreteErrorModel result;
    // Bilinear (Tustin) discretization. Solve all eight right-hand sides in
    // one pivoted elimination: (I-T*A/2)^-1 [I+T*A/2, T*B, T*c].
    std::array<std::array<double, 14>, 6> augmented{};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            const double identity = i == j ? 1.0 : 0.0;
            augmented[i][j] = identity - 0.5 * dt * jac[i][j];
            augmented[i][6 + j] = identity + 0.5 * dt * jac[i][j];
        }
        augmented[i][12] = dt * input[i];
        augmented[i][13] = dt * (f0[i] - linear_part[i] - input[i] * steer);
    }
    for (int col = 0; col < 6; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 6; ++row)
            if (std::abs(augmented[row][col]) > std::abs(augmented[pivot][col]))
                pivot = row;
        if (!std::isfinite(augmented[pivot][col]) ||
            std::abs(augmented[pivot][col]) < 1e-12)
            throw std::runtime_error("Three-DOF Tustin discretization is singular");
        if (pivot != col) std::swap(augmented[pivot], augmented[col]);
        const double diagonal = augmented[col][col];
        for (int j = col; j < 14; ++j) augmented[col][j] /= diagonal;
        for (int row = 0; row < 6; ++row) {
            if (row == col) continue;
            const double factor = augmented[row][col];
            for (int j = col; j < 14; ++j)
                augmented[row][j] -= factor * augmented[col][j];
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) result.a[i][j] = augmented[i][6 + j];
        result.b[i] = augmented[i][12];
        result.c[i] = augmented[i][13];
    }
    return result;
}
