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

LateralMpc::LateralMpc(const VehicleParameters& v,
                       const MpcParameters& p,
                       const DoubleLaneChange& r,
                       const TireTable& tires)
    : vehicle_(v), p_(p), ref_(r), tires_(tires) {
    if (p.np < 2 || p.np > 150 || p.nc < 1 || p.nc > p.np ||
        p.dt <= 0.0 || !std::isfinite(p.max_steer) || p.max_steer <= 0.0 ||
        !(p.max_steer_rate > 0.0) || (p.speed_rate_enabled && !std::isfinite(p.max_steer_rate)) ||
        !std::isfinite(p.speed_rate_start) || !std::isfinite(p.speed_rate_end) ||
        p.speed_rate_start < 0 || p.speed_rate_end <= p.speed_rate_start ||
        !std::isfinite(p.speed_rate_end_limit) || p.speed_rate_end_limit <= 0 ||
        p.admm_rho <= 0.0 || p.admm_max_iterations <= 0 ||
        p.admm_primal_tol <= 0.0 || p.admm_dual_tol <= 0.0 ||
        !std::isfinite(p.r_delta_change) || p.r_delta_change < 0.0 ||
        !std::isfinite(p.reference_curvature_sigma_m) || p.reference_curvature_sigma_m < 0.0 ||
        !std::isfinite(p.tire_slope_floor) || p.tire_slope_floor < 0.0 || p.tire_slope_floor > 1.0 ||
        !std::isfinite(p.front_relaxation_m) || p.front_relaxation_m < 0.0 ||
        !std::isfinite(p.rear_relaxation_m) || p.rear_relaxation_m < 0.0 ||
        !std::isfinite(p.stage_weight_lambda) || std::abs(p.stage_weight_lambda) > 5.0)
        throw std::invalid_argument("Invalid MPC settings (require 1 <= Nc <= Np <= 150)");
    if (!std::isfinite(v.road_grip_scale) || v.road_grip_scale <= 0.0)
        throw std::invalid_argument("Invalid road grip");
    current_steer_rate_ = p.max_steer_rate;
    last_solution_.assign(p.nc, 0.0);
}

void LateralMpc::setRoadGripProfile(std::function<double(double)> grip_at_x) {
    road_grip_at_x_ = std::move(grip_at_x);
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
    nominal_states_.assign(np + 1, PredictionState{});
    nominal_states_[0] = {plant.vy, plant.vx, plant.psi, plant.r,
                           plant.y, plant.x, plant.alpha_front, plant.alpha_rear};
    sensitivity_.assign(np, std::vector<PredictionState>(nc, PredictionState{}));
    std::vector<PredictionState> current(nc, PredictionState{});
    const auto [front_load, rear_load] = staticWheelLoads(vehicle_);
    const double mu = road_grip_at_x_ ? road_grip_at_x_(plant.x) : vehicle_.road_grip_scale;
    // Anchor the first prediction to the observed forces. Fade the residual
    // as the forecast moves away from the measured load and tire state.
    const auto measured_bias = [&](double force, double slip, double load, double length) {
        if (length == 0.0 || !std::isfinite(force)) return 0.0;
        return force - 2.0 * mu * tires_.boundedLateralForce(slip / mu, load);
    };
    const double front_bias = measured_bias(plant.fy_front, plant.alpha_front, front_load,
                                            p_.front_relaxation_m);
    const double rear_bias = measured_bias(plant.fy_rear, plant.alpha_rear, rear_load,
                                           p_.rear_relaxation_m);
    for (int stage = 0; stage < np; ++stage) {
        const int active = std::min(stage, nc - 1);
        const auto& state = nominal_states_[stage];
        VehicleState operating;
        operating.vy = state[0]; operating.vx = state[1];
        operating.psi = state[2]; operating.r = state[3];
        operating.y = state[4]; operating.x = state[5];
        operating.alpha_front = state[6]; operating.alpha_rear = state[7];
        // A measured acceleration is a local disturbance, not a constant
        // full-horizon braking command. Fade it during the nominal rollout.
        const double stage_ax = ax * std::exp(-stage * p_.dt / 0.5);
        const double force_bias_scale = std::exp(-stage * p_.dt / 0.08);
        VehicleParameters stage_vehicle = vehicle_;
        if (road_grip_at_x_) stage_vehicle.road_grip_scale = road_grip_at_x_(operating.x);
        const auto model = discretizeErrorModel(stage_vehicle, tires_, operating,
                                                nominal_steering_[active], stage_ax, p_.dt,
                                                p_.tire_slope_floor,
                                                p_.front_relaxation_m, p_.rear_relaxation_m,
                                                front_bias * force_bias_scale,
                                                rear_bias * force_bias_scale);
        auto next_state = matrixTimesState(model.a, state);
        for (int d = 0; d < prediction_state_count; ++d)
            next_state[d] += model.b[d] * nominal_steering_[active] + model.c[d];
        for (double value : next_state)
            if (!std::isfinite(value))
                throw std::runtime_error("LTV prediction became nonfinite");
        nominal_states_[stage + 1] = next_state;
        std::vector<PredictionState> next(nc, PredictionState{});
        for (int j = 0; j < nc; ++j) {
            next[j] = matrixTimesState(model.a, current[j]);
            // Delta-steering j changes every later absolute steering command.
            if (j <= active)
                for (int d = 0; d < prediction_state_count; ++d) next[j][d] += model.b[d];
        }
        current = std::move(next);
        sensitivity_[stage] = current;
    }
    low_.assign(2*nc,0.0);
    high_.assign(2*nc,0.0);
}
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

double LateralMpc::predictedSteeringAt(int prediction_step) const {
    if (prediction_step < 0 || prediction_step >= p_.np)
        throw std::out_of_range("Prediction step is outside [0,Np)");
    return last_solution_[std::min(prediction_step, p_.nc - 1)];
}

MpcResult LateralMpc::control(const VehicleState& plant, double prev, double previous_increment) {
    if (!std::isfinite(prev) || std::abs(prev) > p_.max_steer)
        throw std::invalid_argument("Previous steering is outside the MPC angle bound");
    try {
        return solve(plant, prev, previous_increment);
    } catch (const std::exception& e) {
        has_solution_ = false;
        MpcResult status;
        status.steer = prev;
        status.prediction_failure = e.what();
        return status;
    }
}

MpcResult LateralMpc::solve(const VehicleState& plant, double prev, double previous_increment) {
    if (!std::isfinite(plant.vx) || !std::isfinite(plant.ax))
        throw std::runtime_error("MPC requires finite vehicle speed and acceleration");
    current_steer_rate_ = effectiveSteerRate(p_, plant.vx);
    precompute(plant, prev, std::clamp(plant.ax, -6.0, 6.0));
    const int np = p_.np, nc = p_.nc;
    std::vector<double> gradient(nc, 0.0);
    hessian_.assign(nc, std::vector<double>(nc, 0.0));
    const double q[4] = {p_.q_y, p_.q_psi, p_.q_vy, p_.q_r};
    const double wheelbase = vehicle_.a + vehicle_.b;
    const auto [front_load, rear_load] = staticWheelLoads(vehicle_);
    const bool use_feedforward = p_.steer_feedforward && p_.r_steer > 0.0;
    const double peak_accel = use_feedforward ? 2.0 *
        (tires_.peakLateralForce(front_load) + tires_.peakLateralForce(rear_load)) /
        vehicle_.mass : 0.0;
    const double understeer_per_vx2 = use_feedforward ? vehicle_.mass / (2.0 * wheelbase) *
        (vehicle_.b / tires_.corneringStiffness(front_load) -
         vehicle_.a / tires_.corneringStiffness(rear_load)) : 0.0;
    for (int stage = 0; stage < np; ++stage) {
        const auto& nominal = nominal_states_[stage + 1];
        const auto path = ref_.at(nominal[5]);
        const double heading_slope = path.d2ydx2 /
            (1.0 + path.dydx * path.dydx);
        const auto ahead = ref_.at(nominal[5] + 0.2);
        const auto behind = ref_.at(nominal[5] - 0.2);
        const double heading_slope_x =
            (ahead.d2ydx2 / (1.0 + ahead.dydx * ahead.dydx) -
             behind.d2ydx2 / (1.0 + behind.dydx * behind.dydx)) / 0.4;
        const double cpsi = std::cos(nominal[2]);
        const double spsi = std::sin(nominal[2]);
        const double x_dot = nominal[1] * cpsi - nominal[0] * spsi;
        const double y_dot = nominal[1] * spsi + nominal[0] * cpsi;
        const double ey_dot = y_dot - path.dydx * x_dot;
        const double heading_rate = heading_slope * x_dot;
        // With tire sideslip, the CG velocity direction can follow the path
        // tangent while the body yaw differs from it.
        const double course_offset = p_.track_course
            ? std::atan2(nominal[0], nominal[1]) : 0.0;
        const double speed_squared = std::max(1e-6,
            nominal[0] * nominal[0] + nominal[1] * nominal[1]);
        const double error[4] = {nominal[4] - path.y,
            angleDifference(nominal[2] + course_offset, path.psi), ey_dot,
            nominal[3] - heading_rate};
        std::vector<std::array<double, 4>> error_sensitivity(nc);
        for (int j = 0; j < nc; ++j) {
            const auto& s = sensitivity_[stage][j];
            const double x_dot_s = cpsi * s[1] - spsi * s[0] - y_dot * s[2];
            const double y_dot_s = spsi * s[1] + cpsi * s[0] + x_dot * s[2];
            const double course_offset_s = p_.track_course
                ? (nominal[1] * s[0] - nominal[0] * s[1]) / speed_squared : 0.0;
            error_sensitivity[j] = {s[4] - path.dydx * s[5],
                s[2] + course_offset_s - heading_slope * s[5],
                y_dot_s - path.dydx * x_dot_s - path.d2ydx2 * x_dot * s[5],
                s[3] - heading_slope * x_dot_s - heading_slope_x * x_dot * s[5]};
        }
        double error_offset[4] = {error[0], error[1], error[2], error[3]};
        for (int j = 0; j < nc; ++j)
            for (int d = 0; d < 4; ++d)
                error_offset[d] -= error_sensitivity[j][d] *
                                   nominal_increments_[j];
        const double terminal = (stage == np - 1 ? p_.terminal_scale : 1.0) *
            std::exp(p_.stage_weight_lambda * (stage + 1.0) / np);
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
        double ff = 0.0;
        if (use_feedforward) {
            const double grip = road_grip_at_x_ ? road_grip_at_x_(nominal[5]) : vehicle_.road_grip_scale;
            const double vx2 = nominal[1] * nominal[1];
            const double curve_limit = grip * peak_accel / std::max(1.0, vx2);
            const double stiffness_scale = p_.usesTireRelaxation() ? 1.0 : grip;
            const double understeer = understeer_per_vx2 * vx2 / stiffness_scale;
            const double curvature = ref_.controlCurvature(nominal[5], p_.reference_curvature_sigma_m);
            ff = std::clamp((wheelbase + understeer) *
                std::clamp(curvature, -curve_limit, curve_limit), -p_.max_steer, p_.max_steer);
        }
        // Absolute steering = prev + prefix sum of optimized increments.
        for (int j = 0; j <= active; ++j) {
            gradient[j] += 2.0 * p_.r_steer * (prev - ff);
            for (int k = 0; k <= active; ++k)
                hessian_[j][k] += 2.0 * p_.r_steer;
        }
    }
    for (int i = 0; i < nc; ++i)
        hessian_[i][i] += 2.0 * p_.r_delta + 1e-9;
    // Penalize sudden changes in steering rate, starting from the last applied move.
    const double rate_change = 2.0 * p_.r_delta_change;
    gradient[0] -= rate_change * previous_increment;
    hessian_[0][0] += rate_change;
    for (int i = 1; i < nc; ++i) {
        hessian_[i][i] += rate_change;
        hessian_[i - 1][i - 1] += rate_change;
        hessian_[i][i - 1] -= rate_change;
        hessian_[i - 1][i] -= rate_change;
    }
    if (nc < np) hessian_[nc - 1][nc - 1] += rate_change;
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
        for (double move : u)
            if (!std::isfinite(move))
                throw std::runtime_error("QP produced a nonfinite steering increment");

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
    has_solution_ = true;
    return status;
}
