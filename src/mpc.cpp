#include "mpc.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

double effectiveSteerRate(const MpcParameters& p, double vx) {
    if (!p.speed_rate_enabled) return p.max_steer_rate;
    const double fraction = std::clamp((std::abs(vx) - p.speed_rate_start) /
        (p.speed_rate_end - p.speed_rate_start), 0.0, 1.0);
    return p.max_steer_rate + fraction *
        (std::min(p.max_steer_rate, p.speed_rate_end_limit) - p.max_steer_rate);
}

bool optimizationFailed(const MpcResult& result) {
    return (result.iterations > 0 && !result.converged) || !result.prediction_failure.empty();
}

LateralMpc::LateralMpc(const VehicleParameters& v,
                       const MpcParameters& p,
                       const DoubleLaneChange& r,
                       const TireTable& tires)
    : vehicle_(v), p_(p), ref_(r), tires_(tires) {
    if (p.np < 2 || p.np > 150 || p.nc < 1 || p.nc > p.np ||
        p.dt <= 0.0 || p.max_steer <= 0.0 || p.max_steer_rate <= 0.0 ||
        !std::isfinite(p.speed_rate_start) || !std::isfinite(p.speed_rate_end) ||
        p.speed_rate_start < 0 || p.speed_rate_end <= p.speed_rate_start ||
        !std::isfinite(p.speed_rate_end_limit) || p.speed_rate_end_limit <= 0 ||
        p.admm_rho <= 0.0 || p.admm_max_iterations <= 0 ||
        p.admm_primal_tol <= 0.0 || p.admm_dual_tol <= 0.0 ||
        !std::isfinite(p.grip_preview_s) || p.grip_preview_s <= 0.0 ||
        !std::isfinite(p.tire_slope_floor) || p.tire_slope_floor < 0.0 || p.tire_slope_floor > 1.0)
        throw std::invalid_argument("Invalid MPC settings (require 1 <= Nc <= Np <= 150)");
    setRoadGrip(v.road_grip_scale);
    current_steer_rate_ = p.max_steer_rate;
    last_solution_.assign(p.nc, 0.0);
}

void LateralMpc::setRoadGrip(double grip) {
    if (!std::isfinite(grip) || grip <= 0.0) throw std::invalid_argument("Invalid road grip");
    vehicle_.road_grip_scale = grip;
    const auto [front_load, rear_load] = staticWheelLoads(vehicle_);
    front_stiffness_ = 2.0 * grip * tires_.corneringStiffness(front_load);
    rear_stiffness_ = 2.0 * grip * tires_.corneringStiffness(rear_load);
    max_lateral_accel_ = 2.0 * grip *
        (tires_.peakLateralForce(front_load) + tires_.peakLateralForce(rear_load)) / vehicle_.mass;
}

void LateralMpc::adoptPreviousSolution(const LateralMpc& previous) {
    if (previous.last_solution_.size() != last_solution_.size()) {
        has_solution_ = false;
        return;
    }
    last_solution_ = previous.last_solution_;
    has_solution_ = previous.has_solution_;
}

void LateralMpc::precompute(const VehicleState& plant, double prev, double ax) {
    const int np = p_.np, nc = p_.nc;
    nominal_steering_.resize(nc);
    nominal_increments_.resize(nc);
    const double max_step = p_.dt * current_steer_rate_;
    double last = prev;
    for (int i = 0; i < nc; ++i) {
        const double candidate = has_solution_
            ? last_solution_[std::min(i + 1, nc - 1)] : prev;
        nominal_steering_[i] = std::clamp(candidate,
            std::max(-p_.max_steer, last - max_step),
            std::min(p_.max_steer, last + max_step));
        nominal_increments_[i] = nominal_steering_[i] - last;
        last = nominal_steering_[i];
    }
    nominal_states_.assign(np + 1, State6{});
    nominal_states_[0] = {plant.vy, plant.vx, plant.psi, plant.r,
                           plant.y, plant.x};
    sensitivity_.assign(np, std::vector<State6>(nc, State6{}));
    std::vector<State6> current(nc, State6{});
    for (int stage = 0; stage < np; ++stage) {
        const int active = std::min(stage, nc - 1);
        const auto& state = nominal_states_[stage];
        VehicleState operating;
        operating.vy = state[0]; operating.vx = state[1];
        operating.psi = state[2]; operating.r = state[3];
        operating.y = state[4]; operating.x = state[5];
        // A measured acceleration is a local disturbance, not a constant
        // full-horizon braking command. Fade it during the nominal rollout.
        const double stage_ax = ax * std::exp(-stage * p_.dt / 0.5);
        const auto model = discretizeErrorModel(vehicle_, tires_, operating,
                                                nominal_steering_[active], stage_ax, p_.dt, p_.nonlinear_tire,
                                                p_.tire_slope_floor);
        auto next_state = matrixTimesState(model.a, state);
        for (int d = 0; d < 6; ++d)
            next_state[d] += model.b[d] * nominal_steering_[active] + model.c[d];
        for (double value : next_state)
            if (!std::isfinite(value))
                throw std::runtime_error("LTV prediction became nonfinite");
        nominal_states_[stage + 1] = next_state;
        std::vector<State6> next(nc, State6{});
        for (int j = 0; j < nc; ++j) {
            next[j] = matrixTimesState(model.a, current[j]);
            // Delta-steering j changes every later absolute steering command.
            if (j <= active)
                for (int d = 0; d < 6; ++d) next[j][d] += model.b[d];
        }
        current = std::move(next);
        sensitivity_[stage] = current;
    }
    low_.assign(2*nc,0.0);
    high_.assign(2*nc,0.0);
}
//
std::vector<std::vector<double>> LateralMpc::choleskyFactor(
    const std::vector<std::vector<double>>& a) {
    const int n = static_cast<int>(a.size());
    std::vector<std::vector<double>> l(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double v = a[i][j];
            for (int k = 0; k < j; ++k) v -= l[i][k] * l[j][k];
            if (i == j) {
                if (!(v > 0.0)) throw std::runtime_error("QP matrix not positive definite");
                l[i][j] = std::sqrt(v);
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    return l;
}

//
std::vector<double> LateralMpc::solveCholesky(const std::vector<double>& rhs) const {
    const int n = p_.nc;
    std::vector<double> y(n, 0.0), x(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = rhs[i];
        for (int j = 0; j < i; ++j) s -= cholesky_[i][j] * y[j];
        y[i] = s / cholesky_[i][i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < n; ++j) s -= cholesky_[j][i] * x[j];
        x[i] = s / cholesky_[i][i];
    }
    return x;
}

//
void LateralMpc::buildBounds(double prev) {
    const int nc = p_.nc;
    const double step = current_steer_rate_ * p_.dt;
    for (int i = 0; i < nc; ++i) {
        // G=[cumulative-sum; identity] acts on steering increments.
        low_[i] = -p_.max_steer - prev;
        high_[i] = p_.max_steer - prev;
        low_[nc + i] = -step;
        high_[nc + i] = step;
    }
}

//
std::vector<double> LateralMpc::constraintProduct(const std::vector<double>& u) const {
    const int nc = p_.nc;
    std::vector<double> v(2 * nc, 0.0);
    double cumulative = 0.0;
    for (int i = 0; i < nc; ++i) {
        cumulative += u[i];
        v[i] = cumulative;
        v[nc + i] = u[i];
    }
    return v;
}

//
std::vector<double> LateralMpc::constraintTranspose(const std::vector<double>& v) const {
    const int nc = p_.nc;
    std::vector<double> u(nc, 0.0);
    double suffix = 0.0;
    for (int i = nc - 1; i >= 0; --i) {
        suffix += v[i];
        u[i] = suffix + v[nc + i];
    }
    return u;
}

//
double LateralMpc::predictedSteeringAt(int prediction_step) const {
    if (prediction_step < 0 || prediction_step >= p_.np)
        throw std::out_of_range("Prediction step is outside [0,Np)");
    return last_solution_[std::min(prediction_step, p_.nc - 1)];
}

MpcResult LateralMpc::gripLimitedControl(const VehicleState& plant,
                                        double previous_steer) const {
    const double speed = std::max(1.0, std::abs(plant.vx));
    const double lookahead = std::clamp(p_.grip_preview_s * speed, 10.0, 28.0);
    const auto target = ref_.at(plant.x + lookahead);
    const double dx = lookahead;
    const double dy = target.y - plant.y;
    const double distance = std::max(1.0, std::hypot(dx, dy));
    const double bearing_error = angleDifference(std::atan2(dy, dx), plant.psi);
    const double max_curvature = max_lateral_accel_ / (speed * speed);
    const double desired_curvature = std::clamp(
        2.0 * std::sin(bearing_error) / distance, -max_curvature, max_curvature);
    const double desired_yaw_rate = speed * desired_curvature;
    const double wheelbase = vehicle_.a + vehicle_.b;
    const double understeer = vehicle_.mass * speed * speed / wheelbase *
        (vehicle_.b / front_stiffness_ - vehicle_.a / rear_stiffness_);
    const double sideslip = std::atan2(plant.vy, speed);
    const double desired_steer = (wheelbase + understeer) * desired_curvature
        + 0.15 * (desired_yaw_rate - plant.r) - 0.35 * sideslip;
    const double max_step = p_.dt * current_steer_rate_;
    MpcResult result;
    result.steer = std::clamp(desired_steer,
        std::max(-p_.max_steer, previous_steer - max_step),
        std::min(p_.max_steer, previous_steer + max_step));
    result.grip_limited = true;
    return result;
}

MpcResult LateralMpc::enterGripLimitedMode(const VehicleState& plant,
                                          double previous_steer) {
    current_steer_rate_ = effectiveSteerRate(p_, plant.vx);
    grip_mode_ = true;
    has_solution_ = false;
    return gripLimitedControl(plant, previous_steer);
}

//
MpcResult LateralMpc::control(const VehicleState& plant, double prev) {
    if (!std::isfinite(plant.vx) || !std::isfinite(plant.ax))
        throw std::runtime_error("MPC requires finite vehicle speed and longitudinal acceleration");
    current_steer_rate_ = effectiveSteerRate(p_, plant.vx);
    const auto current_ref = ref_.at(plant.x);
    const double position_error = plant.y - current_ref.y;
    const double heading_error = angleDifference(plant.psi, current_ref.psi);
    const double current_demand = plant.vx * plant.vx * std::abs(current_ref.curvature);
    const bool excessive_sideslip = std::abs(std::atan2(plant.vy, std::max(1.0, std::abs(plant.vx)))) > 0.12;
    const bool state_risk = (std::abs(position_error) > 0.6 && std::abs(plant.vy) > 0.6) || excessive_sideslip;
    const bool legacy_risk = current_demand > 1.10 * max_lateral_accel_ ||
        (std::abs(position_error) > 0.3 && std::abs(plant.vy) > 0.5);
    if (p_.grip_switch_enabled && (p_.grip_state_only ? state_risk : legacy_risk))
        grip_mode_ = true;
    if (grip_mode_ && (p_.grip_state_only || current_demand < 0.7 * max_lateral_accel_) &&
        std::abs(position_error) < 0.08 && std::abs(heading_error) < 0.05 &&
        std::abs(plant.vy) < 0.3)
        grip_mode_ = false;
    if (grip_mode_ || plant.vx < 1.0) {
        has_solution_ = false;
        return gripLimitedControl(plant, prev);
    }
    const double ax = std::clamp(plant.ax, -6.0, 6.0);
    try { precompute(plant, prev, ax); }
    catch (const std::exception& ex) {
        grip_mode_ = true;
        has_solution_ = false;
        auto fallback = gripLimitedControl(plant, prev);
        fallback.prediction_failure = ex.what();
        return fallback;
    }
    const int np = p_.np, nc = p_.nc;
    std::vector<double> gradient(nc, 0.0);
    hessian_.assign(nc, std::vector<double>(nc, 0.0));
    const double q[4] = {p_.q_y, p_.q_psi, p_.q_vy, p_.q_r};
    const double wheelbase = vehicle_.a + vehicle_.b;
    for (int stage = 0; stage < np; ++stage) {
        const auto& nominal = nominal_states_[stage + 1];
        const auto path = ref_.at(nominal[5]);
        const double curve_limit = max_lateral_accel_ /
            std::max(1.0, nominal[1] * nominal[1]);
        const double curvature = std::clamp(path.curvature,
                                            -curve_limit, curve_limit);
        const double heading_slope = path.d2ydx2 /
            (1.0 + path.dydx * path.dydx);
        const double curvature_slope = (ref_.at(nominal[5] + 0.2).curvature -
            ref_.at(nominal[5] - 0.2).curvature) / 0.4;
        const double error[4] = {nominal[4] - path.y,
            angleDifference(nominal[2], path.psi), nominal[0],
            nominal[3] - nominal[1] * curvature};
        std::vector<std::array<double, 4>> error_sensitivity(nc);
        for (int j = 0; j < nc; ++j) {
            const auto& s = sensitivity_[stage][j];
            error_sensitivity[j] = {s[4] - path.dydx * s[5],
                s[2] - heading_slope * s[5], s[0],
                s[3] - curvature * s[1] -
                nominal[1] * curvature_slope * s[5]};
        }
        double error_offset[4] = {error[0], error[1], error[2], error[3]};
        for (int j = 0; j < nc; ++j)
            for (int d = 0; d < 4; ++d)
                error_offset[d] -= error_sensitivity[j][d] *
                                   nominal_increments_[j];
        const double terminal = stage == np - 1 ? p_.terminal_scale : 1.0;
        for (int j = 0; j < nc; ++j) {
            for (int d = 0; d < 4; ++d)
                gradient[j] += 2.0 * terminal * q[d] *
                               error_sensitivity[j][d] * error_offset[d];
            for (int k = 0; k < nc; ++k)
                for (int d = 0; d < 4; ++d)
                    hessian_[j][k] += 2.0 * terminal * q[d] *
                        error_sensitivity[j][d] * error_sensitivity[k][d];
        }
        const int active = std::min(stage, nc - 1);
        const double understeer = vehicle_.mass * nominal[1] * nominal[1] /
            wheelbase * (vehicle_.b / front_stiffness_ -
                         vehicle_.a / rear_stiffness_);
        const double ff = std::clamp((wheelbase + understeer) * curvature,
                                     -p_.max_steer, p_.max_steer);
        // Absolute steering = prev + prefix sum of optimized increments.
        for (int j = 0; j <= active; ++j) {
            gradient[j] += 2.0 * p_.r_steer * (prev - ff);
            for (int k = 0; k <= active; ++k)
                hessian_[j][k] += 2.0 * p_.r_steer;
        }
    }
    for (int i = 0; i < nc; ++i)
        hessian_[i][i] += 2.0 * p_.r_delta + 1e-9;
    auto lhs = hessian_;
    // G=[lower-triangular prefix sums; identity].
    for (int j = 0; j < nc; ++j)
        for (int k = 0; k < nc; ++k)
            lhs[j][k] += p_.admm_rho *
                (nc - std::max(j, k) + (j == k ? 1.0 : 0.0));
    cholesky_ = choleskyFactor(lhs);
    buildBounds(prev);

    std::vector<double> u = nominal_increments_;
    auto z = constraintProduct(u);
    std::vector<double> dual(2 * nc, 0.0);

	// ADMM iterations
    MpcResult status;
    for (int iteration = 0; iteration < p_.admm_max_iterations; ++iteration) {
        std::vector<double> z_minus_dual(2 * nc);
        for (int i = 0; i < 2 * nc; ++i) z_minus_dual[i] = z[i] - dual[i];
        auto rhs_correction = constraintTranspose(z_minus_dual);
        std::vector<double> rhs(nc);
        for (int i = 0; i < nc; ++i)
            rhs[i] = -gradient[i] + p_.admm_rho * rhs_correction[i];
        u = solveCholesky(rhs);

        auto au = constraintProduct(u);
        auto z_old = z;
        double primal_inf = 0.0;
        for (int i = 0; i < 2 * nc; ++i) {
            z[i] = std::clamp(au[i] + dual[i], low_[i], high_[i]);
            dual[i] += au[i] - z[i];
            primal_inf = std::max(primal_inf, std::abs(au[i] - z[i]));
        }
        for (int i = 0; i < 2 * nc; ++i) z_old[i] = z[i] - z_old[i];
        auto dual_change = constraintTranspose(z_old);
        double dual_inf = 0.0;
        for (double v : dual_change)
            dual_inf = std::max(dual_inf, std::abs(p_.admm_rho * v));
        status.iterations = iteration + 1;
        status.primal_residual = primal_inf;
        status.dual_residual = dual_inf;
        if (primal_inf < p_.admm_primal_tol && dual_inf < p_.admm_dual_tol) {
            status.converged = true;
            break;
        }
    }

    // Last-resort projection protects actuator limits even if solver did not converge.
    // Nonconvergence must be checked using status.converged in experimental results.
    double last = prev;
    const double max_step = p_.dt * current_steer_rate_;
    for (int i = 0; i < nc; ++i) {
        last = std::clamp(last + u[i],
            std::max(-p_.max_steer, last - max_step),
            std::min(p_.max_steer, last + max_step));
        last_solution_[i] = last;
    }
    status.steer = last_solution_[0];
    if (!status.converged) {
        grip_mode_ = true;
        has_solution_ = false;
        auto fallback = gripLimitedControl(plant, prev);
        fallback.iterations = status.iterations;
        fallback.primal_residual = status.primal_residual;
        fallback.dual_residual = status.dual_residual;
        return fallback;
    }
    has_solution_ = true;
    return status;
}
