#include "rl.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace {
double clip(double x, double lo, double hi) { return std::clamp(x, lo, hi); }

double neutralWeight(const RlConfig& config, std::size_t k) {
    const auto& choices = config.weight_choices[k];
    return choices.empty() ? std::sqrt(config.weight_min[k] * config.weight_max[k])
                           : choices[choices.size() / 2];
}

}

double lateralErrorRate(const VehicleState& state, const ReferencePoint& reference) {
    const double c = std::cos(state.psi), s = std::sin(state.psi);
    const double x_dot = state.vx * c - state.vy * s;
    const double y_dot = state.vx * s + state.vy * c;
    return y_dot - reference.dydx * x_dot;
}

double scalarizedRlReward(double ey, double epsi, double steer, int np,
                          const MpcParameters& mpc, const RlConfig& config,
                          double previous_steer, int previous_np,
                          const MpcTuningAction& action,
                          const MpcTuningAction& previous_action,
                          double vy, double yaw_error, double sideslip, double ey_dot) {
    const auto& r = config.reward;
    if (r.mode == "joint_tracking") {
        const double heading = angleDifference(epsi, 0.0);
        const double course = angleDifference(heading + sideslip, 0.0);
        const double ey_loss = r.w_y * std::pow(clip(ey / r.ey_scale_m, -20.0, 20.0), 2);
        const double heading_loss = r.w_heading * std::pow(clip(heading / r.heading_scale_rad, -20.0, 20.0), 2);
        const double course_loss = r.w_course * std::pow(clip(course / r.heading_scale_rad, -20.0, 20.0), 2);
        const double steer_rate = (steer - previous_steer) / (r.steer_rate_scale_rad_s * mpc.dt);
        const double slip_excess = std::max(0.0, std::abs(sideslip) / r.sideslip_threshold_rad - 1.0);
        const double peak_excess = std::max(0.0, std::abs(ey) - r.peak_threshold_m) / r.peak_threshold_m;
        const double loss = ey_loss + heading_loss + course_loss +
            r.w_peak * peak_excess * peak_excess +
            r.w_steer * std::pow(steer / mpc.max_steer, 2) +
            r.w_steer_rate * std::pow(clip(steer_rate, -5.0, 5.0), 2) +
            r.w_sideslip * slip_excess * slip_excess;
        return -r.reward_scale * loss;
    }
    const double normalized_ey = ey / r.ey_scale_m;
    if (r.tracking_v2 && mpc.track_course)
        epsi = angleDifference(epsi + sideslip, 0.0);
    const double normalized_epsi = clip(epsi / r.epsi_scale_rad, -20.0, 20.0);
    const double peak_excess = std::max(0.0, std::abs(ey) - r.peak_threshold_m) /
                               r.peak_threshold_m;
    // Smooth absolute tracking error: the paper's normalized |e| term with a
    // finite derivative at zero. L(1)=1; large errors retain a linear signal.
    const auto tracking_loss = [](double error) {
        return (std::sqrt(1.0 + error * error) - 1.0) /
               (std::sqrt(2.0) - 1.0);
    };
    const double normalized_steer = clip(steer / mpc.max_steer, -1.0, 1.0);
    const double steer_rate_scale = r.tracking_v2 ? r.steer_rate_scale_rad_s : mpc.max_steer_rate;
    const double normalized_steer_rate = std::abs(steer - previous_steer) /
        (steer_rate_scale * mpc.dt);
    const double bounded_steer_rate = std::min(normalized_steer_rate, 2.0);
    // The linear tail still distinguishes large spikes without a quadratic outlier.
    const double steer_rate_loss = bounded_steer_rate * bounded_steer_rate +
        (r.tracking_v2 ? 4.0 * (normalized_steer_rate - bounded_steer_rate) : 0.0);
    const double sideslip_excess = r.tracking_v2
        ? std::max(0.0, std::abs(sideslip) / r.sideslip_threshold_rad - 1.0) : 0.0;
    const double horizon_fraction = clip((np - config.min_np) /
        static_cast<double>(config.max_np - config.min_np), 0.0, 1.0);
    const double horizon_change = previous_np < config.min_np ? 0.0 :
        clip(std::abs(np - previous_np) /
             static_cast<double>(config.max_np - config.min_np), 0.0, 1.0);
    const std::array<double, 3> weights{action.q_y, action.q_psi, action.r_delta};
    const std::array<double, 3> old_weights{previous_action.q_y, previous_action.q_psi,
        previous_action.r_delta};
    constexpr std::array<std::size_t, 3> weight_indices{0, 1, 5};
    double deviation = 0.0, change = 0.0;
    // Only the three learned Q/R weights contribute to action regularization.
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const auto k = weight_indices[i];
        const double center = neutralWeight(config, k);
        const double log_span = 0.5 * std::log(config.maxWeight(k) / config.minWeight(k));
        const double log_weight = clip(std::log(weights[i] / center) / log_span, -1.0, 1.0);
        deviation += log_weight * log_weight / 3.0;
        const double log_change = clip(std::log(weights[i] / old_weights[i]) / log_span, -2.0, 2.0);
        change += log_change * log_change / 3.0;
    }
    const double lateral_loss = r.squared_lateral_reward
        ? normalized_ey * normalized_ey
        : tracking_loss(clip(normalized_ey, -20.0, 20.0));
    // Penalize a predicted increase in absolute error, including crossing the line too fast.
    const double growth = std::max(0.0, std::abs(ey + r.lateral_growth_preview_s * ey_dot)
        - std::abs(ey)) / r.ey_scale_m;
    const std::array<double, 12> terms{{r.w_y * lateral_loss,
        r.w_peak * peak_excess * peak_excess, r.w_psi * tracking_loss(normalized_epsi),
        r.w_lateral_growth * growth * growth,
        r.w_steer * normalized_steer * normalized_steer,
        r.w_steer_rate * steer_rate_loss,
        r.w_sideslip * sideslip_excess * sideslip_excess,
        r.w_horizon * horizon_fraction + r.w_np_change * horizon_change * horizon_change,
        r.w_vy * std::pow(clip(vy / 2.0, -3.0, 3.0), 2),
        r.w_yaw_error * std::pow(clip(yaw_error / 0.25, -3.0, 3.0), 2),
        r.w_weight_change * change, r.w_weight_deviation * deviation}};
    const double loss = std::accumulate(terms.begin(), terms.end(), 0.0);
    if (r.squared_lateral_reward)
        return -r.reward_scale * loss;
    // Zero is the best possible reward. Smaller losses move a negative return
    // toward zero; log1p keeps a useful signal without clipping large errors.
    return -4.0 * r.reward_scale * std::log1p(loss);
}

double rlDecisionReward(double step_reward_sum, int steps, int qp_failures,
                        bool failed, bool incomplete, const RlConfig& config) {
    const auto& r = config.reward;
    const int penalized_qp = config.penalize_qp_failure ? qp_failures : 0;
    if (r.tracking_v2 || r.mode == "joint_tracking") {
        const int valid_steps = steps - static_cast<int>(failed);
        const double mean = step_reward_sum / (std::max(1, valid_steps) * r.reward_scale);
        const double terminal = failed ? r.failure_penalty
            : (incomplete ? r.incomplete_episode_penalty : 0.0);
        return mean - penalized_qp * r.qp_failure_penalty - terminal;
    }
    const double penalties = (penalized_qp + static_cast<int>(failed)) * r.failure_penalty +
        (incomplete ? r.incomplete_episode_penalty : 0.0);
    return (step_reward_sum / r.reward_scale - penalties) / std::max(1, steps);
}

MpcTuningAction neutralRlAction(const RlConfig& config) {
    MpcTuningAction action;
    action.np = config.nominal_np;
    action.q_y = neutralWeight(config, 0);
    action.q_psi = neutralWeight(config, 1);
    action.q_vy = neutralWeight(config, 2);
    action.q_r = neutralWeight(config, 3);
    action.r_steer = neutralWeight(config, 4);
    action.r_delta = neutralWeight(config, 5);
    return action;
}

MpcTuningAction advanceRlTuning(const std::array<double, 4>& action,
                               const MpcTuningAction& previous,
                               const RlConfig& config) {
    for (double value : action)
        if (!std::isfinite(value) || value < -1.0 || value > 1.0)
            throw std::invalid_argument("STEP actions must be finite values in [-1,1]");

    auto next = previous;
    const double np_action = action[0];
    const int extent = np_action >= 0.0 ? config.max_np - config.nominal_np
                                       : config.nominal_np - config.min_np;
    const double requested_np = config.nominal_np + np_action * extent;
    const int steps = static_cast<int>(std::lround((requested_np - config.min_np) / config.np_step));
    next.np = config.min_np + steps * config.np_step;
    if (config.max_np_change_per_decision > 0) {
        const int max_change = (config.max_np_change_per_decision / config.np_step) * config.np_step;
        next.np = std::clamp(next.np, previous.np - max_change, previous.np + max_change);
    }

    const double max_step = std::log(config.max_weight_ratio_per_decision);
    const auto weight = [&](std::size_t k, double request, double last) {
        const auto& choices = config.weight_choices[k];
        if (config.weight_action_mode == "legacy" && !choices.empty()) {
            const auto index = std::lround((request + 1.0) * (choices.size() - 1) / 2.0);
            return choices[static_cast<std::size_t>(index)];
        }
        const double center = neutralWeight(config, k);
        const double lo = std::log(config.minWeight(k) / center);
        const double hi = std::log(config.maxWeight(k) / center);
        const double target = std::log(center) + request * (request >= 0.0 ? hi : -lo);
        if (config.weight_action_mode == "continuous_weights")
            return std::exp(clip(target, std::log(last) - max_step, std::log(last) + max_step));
        if (!config.smooth_tuning) return std::exp(target);
        const double current = std::log(last);
        const double error = target - current;
        // Hold the current weight when the request differs by about 0.2% or less.
        if (std::abs(error) <= 0.002) return last;
        return std::exp(current + clip(0.25 * error, -max_step, max_step));
    };
    next.q_y = weight(0, action[1], previous.q_y);
    next.q_psi = weight(1, action[2], previous.q_psi);
    next.r_delta = weight(5, action[3], previous.r_delta);
    return next;
}

double roadPreviewDuration(const RlConfig& config, double mpc_dt) {
    const double decision_s = mpc_dt * config.decision_steps;
    const int slew = (config.max_np_change_per_decision / config.np_step) * config.np_step;
    const int span = config.max_np - config.min_np;
    const int ramp_decisions = slew > 0 ? (span + slew - 1) / slew : 0;
    constexpr double margin_s = 0.30;
    return std::max(config.curvature_preview_s, (ramp_decisions + 1) * decision_s + margin_s);
}

double roadPreviewDistance(const RlConfig& config, double mpc_dt, double x_speed) {
    if (x_speed <= 0.0) return 0.0;
    const double distance = std::max(config.road_preview_min_m,
        x_speed * roadPreviewDuration(config, mpc_dt));
    return config.road_preview_max_m > 0.0
        ? std::min(distance, config.road_preview_max_m) : distance;
}

RoadPreviewObservation makeRoadPreviewObservation(
    const VehicleState& state, const DoubleLaneChange& reference,
    const std::function<double(double)>& grip_at_x, double distance_m,
    const MpcParameters& mpc, const RlConfig& config, int current_np, double previous_steer,
    const MpcTuningAction* executed) {
    constexpr double gravity = 9.80665;
    constexpr double demand_threshold = 1.0;
    constexpr int samples = 32;
    const double speed = std::hypot(state.vx, state.vy);
    const double x_speed = std::max(0.0,
        state.vx * std::cos(state.psi) - state.vy * std::sin(state.psi));
    const double requested_preview_m = roadPreviewDistance(config, mpc.dt, x_speed);
    const auto point = reference.at(state.x);
    const double end_x = std::min(distance_m, point.x_max);
    const double preview_m = std::min(requested_preview_m, std::max(0.0, end_x - state.x));
    const double station_step = preview_m / samples;
    const double grip = grip_at_x(state.x);
    const double curvature = reference.controlCurvature(state.x, mpc.reference_curvature_sigma_m);
    const double near_m = std::min(preview_m, 0.6 * x_speed);
    const double demand_scale = speed * speed / gravity;
    double peak_demand = demand_scale * std::abs(curvature) / grip;
    double near_demand = peak_demand;
    double min_grip = grip;
    double demand_arrival_fraction = peak_demand >= demand_threshold ? 0.0 : 1.0;
    double curvature_change = 0.0;
    double curvature_rate = 0.0;
    double previous_curvature = curvature;
    for (int i = 1; i <= samples && station_step > 0.0; ++i) {
        const double x = state.x + i * station_step;
        const double future = reference.controlCurvature(x, mpc.reference_curvature_sigma_m);
        const double future_grip = grip_at_x(x);
        const double demand = demand_scale * std::abs(future) / future_grip;
        min_grip = std::min(min_grip, future_grip);
        if (i * station_step <= near_m)
            near_demand = std::max(near_demand, demand);
        if (peak_demand < demand_threshold && demand >= demand_threshold)
            demand_arrival_fraction = i * station_step / requested_preview_m;
        peak_demand = std::max(peak_demand, demand);
        if (std::abs(future - curvature) > std::abs(curvature_change))
            curvature_change = future - curvature;
        curvature_rate = std::max(curvature_rate, std::abs(future - previous_curvature) / station_step);
        previous_curvature = future;
    }
    const double near_x = state.x + near_m;
    const double heading_change = angleDifference(reference.at(near_x).psi, point.psi);
    const double course_error = angleDifference(state.psi + vehicleSideslip(state), point.psi);
    RoadPreviewObservation obs{clip(speed / 30.0, 0.0, 2.0),
            clip(curvature * 25.0, -1.0, 1.0),
            clip(curvature_change * 25.0, -1.0, 1.0),
            clip(curvature_rate * 100.0, 0.0, 1.0),
            clip(heading_change / 0.30, -1.0, 1.0),
            clip(grip / 1.5, 0.0, 1.0),
            clip(peak_demand / config.grip_demand_clip, 0.0, 1.0),
            clip(demand_arrival_fraction, 0.0, 1.0),
            clip(std::asinh((state.y - point.y) / config.reward.ey_scale_m), -4.0, 4.0),
            clip(std::asinh(course_error / config.reward.epsi_scale_rad), -4.0, 4.0),
            clip(previous_steer / mpc.max_steer, -1.0, 1.0),
            clip(2.0 * (current_np - config.min_np) /
                static_cast<double>(config.max_np - config.min_np) - 1.0, -1.0, 1.0)};
    if (config.observation_profile == "road_preview_v2" || config.observation_profile == "road_preview_joint") {
        obs.push_back(clip(min_grip / 1.5, 0.0, 1.0));
        obs.push_back(clip(near_demand / config.grip_demand_clip, 0.0, 1.0));
    }
    if (config.observation_profile == "road_preview_joint") {
        const auto& current = executed ? *executed : neutralRlAction(config);
        obs.push_back(clip(angleDifference(state.psi, point.psi) / config.reward.heading_scale_rad, -4.0, 4.0));
        obs.push_back(clip(vehicleSideslip(state) / config.reward.sideslip_threshold_rad, -4.0, 4.0));
        obs.push_back(clip((state.r - state.vx * point.curvature) / 0.25, -4.0, 4.0));
        for (const auto pair : {std::pair<double, std::size_t>{current.q_y, 0}, {current.q_psi, 1}, {current.r_delta, 5}}) {
            const double center = neutralWeight(config, pair.second);
            const double span = std::max(std::log(config.maxWeight(pair.second) / center),
                                         std::log(center / config.minWeight(pair.second)));
            obs.push_back(clip(std::log(pair.first / center) / span, -1.0, 1.0));
        }
    }
    return obs;
}

MpcParameters applyRlAction(const MpcParameters& base, const MpcTuningAction& action,
                            const RlConfig& config) {
    const std::array<double, 6> weights{{action.q_y, action.q_psi,
        action.q_vy, action.q_r, action.r_steer, action.r_delta}};
    for (std::size_t k : {0u, 1u, 5u})
        if (!std::isfinite(weights[k]) || weights[k] < config.minWeight(k) - 1e-9 ||
            weights[k] > config.maxWeight(k) + 1e-9)
            throw std::invalid_argument("RL absolute Q/R weight is outside configured bounds");
    MpcParameters result = base;
    result.np = std::clamp(action.np, std::max(config.min_np, base.nc), config.max_np);
    result.q_y = action.q_y;
    result.q_psi = action.q_psi;
    result.r_delta = action.r_delta;
    return result;
}
