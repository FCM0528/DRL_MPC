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
    if (loads_.front() < 0.0 || !std::is_sorted(loads_.begin(), loads_.end()) ||
        std::adjacent_find(loads_.begin(), loads_.end()) != loads_.end())
        throw std::runtime_error("Tire table loads must be nonnegative and increasing");

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
    if (slip_deg_.size() < 2 || slip_deg_.front() != 0.0 ||
        std::any_of(forces_.front().begin(), forces_.front().end(),
                    [](double force) { return force != 0.0; }))
        throw std::runtime_error("Tire table needs a zero-force row at zero slip");
    if (loads_.front() == 0.0 &&
        std::any_of(forces_.begin(), forces_.end(),
                    [](const auto& row) { return row.front() != 0.0; }))
        throw std::runtime_error("Tire table needs zero force at zero load");
}

double TireTable::positiveForce(double slip_deg, double normal_load_n) const {
    if (!std::isfinite(normal_load_n) || normal_load_n < minLoad() ||
        normal_load_n > maxLoad())
        throw std::invalid_argument("Per-tire normal load is outside tire table range");
    if (!std::isfinite(slip_deg) || slip_deg < 0.0 || slip_deg > slip_deg_.back())
        throw std::invalid_argument("Tire slip angle is outside CarSim tire table range");
    const auto i = lowerIndex(slip_deg_, slip_deg);
    const auto j = lowerIndex(loads_, normal_load_n);
    const double ta = (slip_deg - slip_deg_[i]) / (slip_deg_[i + 1] - slip_deg_[i]);
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

double TireTable::boundedLateralForce(double slip_rad, double normal_load_n) const {
    if (!std::isfinite(slip_rad)) throw std::invalid_argument("Nonfinite tire slip");
    const double slip_deg = std::min(std::abs(slip_rad) * rad2deg, slip_deg_.back());
    return std::copysign(positiveForce(slip_deg, normal_load_n), slip_rad);
}

double TireTable::peakLateralForce(double normal_load_n) const {
    double peak = 0.0;
    for (const double slip : slip_deg_)
        peak = std::max(peak, positiveForce(slip, normal_load_n));
    return peak;
}

PredictionState matrixTimesState(const PredictionMatrix& a, const PredictionState& x) {
    PredictionState y{};
    for (int i = 0; i < prediction_state_count; ++i)
        for (int j = 0; j < prediction_state_count; ++j) y[i] += a[i][j] * x[j];
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
                                        double steer, double ax, double dt,
                                        double tire_slope_floor,
                                        double front_relaxation_m,
                                        double rear_relaxation_m,
                                        double front_force_bias,
                                        double rear_force_bias) {
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
    const bool lag_front = front_relaxation_m > 0.0;
    const bool lag_rear = rear_relaxation_m > 0.0;
    const bool dynamic_tires = lag_front || lag_rear;
    const double front_velocity_ratio = (v + p.a*r) / u;
    const double rear_velocity_ratio = (v - p.b*r) / u;
    const double instant_f = steer - (dynamic_tires ? std::atan(front_velocity_ratio)
                                                     : front_velocity_ratio);
    const double instant_r = -(dynamic_tires ? std::atan(rear_velocity_ratio)
                                              : rear_velocity_ratio);
    const double alpha_f = lag_front ? op.alpha_front : instant_f;
    const double alpha_r = lag_rear ? op.alpha_rear : instant_r;
    // Preserve the measured initial tire stiffness while scaling peak force
    // and the slip at which it occurs with road friction.
    const double slip_scale = dynamic_tires ? p.road_grip_scale : 1.0;
    const auto force = [&](double alpha, double load) {
        return tires.boundedLateralForce(alpha / slip_scale, load);
    };
    const auto slope = [&](double alpha, double load) {
        constexpr double h = 1e-5;
        const double local = (force(alpha + h, load) - force(alpha - h, load)) / (2*h);
        // Preserve control direction where the table is flat or post-peak.
        return std::max(local, tire_slope_floor * tires.corneringStiffness(load) / slip_scale);
    };
    const double cf = 2.0 * p.road_grip_scale * slope(alpha_f, front_load);
    const double cr = 2.0 * p.road_grip_scale * slope(alpha_r, rear_load);
    const double fyf = 2.0 * p.road_grip_scale * force(alpha_f, front_load)
        + front_force_bias;
    const double fyr = 2.0 * p.road_grip_scale * force(alpha_r, rear_load)
        + rear_force_bias;
    const double cpsi = std::cos(yaw), spsi = std::sin(yaw);
    const double lateral_projection = dynamic_tires ? std::cos(steer) : 1.0;
    const double longitudinal_projection = dynamic_tires ? std::sin(steer) : steer;

    PredictionMatrix jac{};
    PredictionState input{};
    PredictionState f0{(fyf*lateral_projection+fyr)/p.mass-u*r, ax, r,
                       (p.a*fyf*lateral_projection-p.b*fyr)/p.iz,
                       u*spsi+v*cpsi, u*cpsi-v*spsi};
    // alpha_dot = vx/L * (alpha_instant - alpha_lagged).
    const auto force_gradient = [&](int index, double axle_x, double length,
                                    double instant, double lagged, double stiffness,
                                    double steer_derivative) {
        PredictionState gradient{};
        const double ratio = (v + axle_x*r) / u;
        const double angle_slope = dynamic_tires ? 1.0 / (1.0 + ratio * ratio) : 1.0;
        gradient[0] = -angle_slope / u;
        gradient[1] = angle_slope * ratio / u;
        gradient[3] = -angle_slope * axle_x / u;
        if (length > 0.0) {
            // Relaxation is strongest near zero slip and drops in saturation.
            // Freeze the slip-dependent length during each LTV linearization.
            constexpr double slip_scale_rad = 2.0 / rad2deg;
            const double slip_ratio = lagged / slip_scale_rad;
            const double effective_length = length / (1.0 + slip_ratio * slip_ratio);
            // Keep the discrete relaxation pole nonnegative for fast tires.
            const double rate = std::min(u / effective_length, 2.0 / dt);
            f0[index] = rate * (instant - lagged);
            for (int j = 0; j < prediction_state_count; ++j)
                jac[index][j] = rate * gradient[j];
            if (rate < 2.0 / dt)
                jac[index][1] += (instant - lagged) / effective_length;
            jac[index][index] = -rate;
            input[index] = rate * steer_derivative;
            gradient = {};
            gradient[index] = stiffness;
        } else {
            for (double& term : gradient) term *= stiffness;
        }
        return gradient;
    };
    const auto front_gradient = force_gradient(6, p.a, front_relaxation_m,
                                                instant_f, alpha_f, cf, 1.0);
    const auto rear_gradient = force_gradient(7, -p.b, rear_relaxation_m,
                                               instant_r, alpha_r, cr, 0.0);
    for (int j = 0; j < prediction_state_count; ++j) {
        const double front_lateral = front_gradient[j] * lateral_projection;
        jac[0][j] = (front_lateral + rear_gradient[j]) / p.mass;
        jac[1][j] = -front_gradient[j] * longitudinal_projection / p.mass;
        jac[3][j] = (p.a * front_lateral - p.b * rear_gradient[j]) / p.iz;
    }
    jac[0][1] -= r;
    jac[0][3] -= u;
    jac[1][0] += r;
    jac[1][3] += v;
    const double force_steer = lag_front ? 0.0 : cf;
    const double lateral_steer = force_steer * lateral_projection -
        (dynamic_tires ? fyf * longitudinal_projection : 0.0);
    input[0] = lateral_steer / p.mass;
    input[1] = -(force_steer * longitudinal_projection + fyf * lateral_projection) / p.mass;
    input[3] = p.a * lateral_steer / p.iz;
    jac[2][3] = 1.0;
    // CarSim uses global X forward and Y left; preserve this rotation.
    jac[4][0] = cpsi;
    jac[4][1] = spsi;
    jac[4][2] = u*cpsi - v*spsi;
    jac[5][0] = -spsi;
    jac[5][1] = cpsi;
    jac[5][2] = -u*spsi - v*cpsi;

    const PredictionState x0{v,u,yaw,r,op.y,op.x,op.alpha_front,op.alpha_rear};
    const auto linear_part = matrixTimesState(jac,x0);
    DiscreteErrorModel result;
    // Bilinear (Tustin) discretization. Solve all right-hand sides in
    // one pivoted elimination: (I-T*A/2)^-1 [I+T*A/2, T*B, T*c].
    constexpr int n = prediction_state_count;
    constexpr int cols = 2 * n + 2;
    std::array<std::array<double, cols>, n> augmented{};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double identity = i == j ? 1.0 : 0.0;
            augmented[i][j] = identity - 0.5 * dt * jac[i][j];
            augmented[i][n + j] = identity + 0.5 * dt * jac[i][j];
        }
        augmented[i][2 * n] = dt * input[i];
        augmented[i][2 * n + 1] = dt * (f0[i] - linear_part[i] - input[i] * steer);
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row)
            if (std::abs(augmented[row][col]) > std::abs(augmented[pivot][col]))
                pivot = row;
        if (!std::isfinite(augmented[pivot][col]) ||
            std::abs(augmented[pivot][col]) < 1e-12)
            throw std::runtime_error("Three-DOF Tustin discretization is singular");
        if (pivot != col) std::swap(augmented[pivot], augmented[col]);
        const double diagonal = augmented[col][col];
        for (int j = col; j < cols; ++j) augmented[col][j] /= diagonal;
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            const double factor = augmented[row][col];
            for (int j = col; j < cols; ++j)
                augmented[row][j] -= factor * augmented[col][j];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) result.a[i][j] = augmented[i][n + j];
        result.b[i] = augmented[i][2 * n];
        result.c[i] = augmented[i][2 * n + 1];
    }
    return result;
}
