#include "rl.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {
double clip(double x, double lo, double hi) { return std::clamp(x, lo, hi); }
}

double scalarizedRlReward(double ey, double epsi, double steer, int np,
                          const MpcParameters& mpc, const RlConfig& config,
                          double previous_steer, int previous_np,
                          const MpcTuningAction& action,
                          const MpcTuningAction& previous_action,
                          double vy, double yaw_error, bool grip_limited,
                          std::array<double, 11>* components) {
    const auto& r = config.reward;
    const double normalized_ey = clip(ey / r.ey_scale_m, -20.0, 20.0);
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
    const double normalized_steer_rate = clip((steer - previous_steer) /
        (mpc.max_steer_rate * mpc.dt), -2.0, 2.0);
    const double horizon_fraction = clip((np - config.min_np) /
        static_cast<double>(config.max_np - config.min_np), 0.0, 1.0);
    const double horizon_change = previous_np < config.min_np ? 0.0 :
        clip(std::abs(np - previous_np) /
             static_cast<double>(config.max_np - config.min_np), 0.0, 1.0);
    const std::array<double, 6> weights{action.q_y, action.q_psi,
        action.q_vy, action.q_r, action.r_steer, action.r_delta};
    const std::array<double, 6> old_weights{previous_action.q_y,
        previous_action.q_psi, previous_action.q_vy,
        previous_action.q_r, previous_action.r_steer,
        previous_action.r_delta};
    double deviation = 0.0, change = 0.0;
    // Only the three learned Q/R weights contribute to action regularization.
    for (std::size_t k : {0u, 1u, 5u}) {
        const double center = std::sqrt(config.weight_min[k] * config.weight_max[k]);
        const double log_span = 0.5 * std::log(config.weight_max[k] / config.weight_min[k]);
        const double log_weight = clip(std::log(weights[k] / center) / log_span, -1.0, 1.0);
        deviation += log_weight * log_weight / 3.0;
        const double log_change = clip(std::log(weights[k] / old_weights[k]) / log_span, -2.0, 2.0);
        change += log_change * log_change / 3.0;
    }
    const std::array<double, 11> terms{{r.w_y * tracking_loss(normalized_ey),
        r.w_peak * peak_excess * peak_excess, r.w_psi * tracking_loss(normalized_epsi),
        r.w_steer * normalized_steer * normalized_steer,
        r.w_steer_rate * normalized_steer_rate * normalized_steer_rate,
        r.w_horizon * horizon_fraction + r.w_np_change * horizon_change * horizon_change,
        r.w_vy * std::pow(clip(vy / 2.0, -3.0, 3.0), 2),
        r.w_yaw_error * std::pow(clip(yaw_error / 0.25, -3.0, 3.0), 2),
        r.w_weight_change * change, r.w_weight_deviation * deviation,
        grip_limited ? r.w_grip_limited : 0.0}};
    if (components) *components = terms;
    const double loss = std::accumulate(terms.begin(), terms.end(), 0.0);
    // Zero is the best possible reward. Smaller losses move a negative return
    // toward zero; log1p keeps a useful signal without clipping large errors.
    return -4.0 * r.reward_scale * std::log1p(loss);
}

MpcTuningAction neutralRlAction(const RlConfig& config) {
    MpcTuningAction action;
    action.np = config.nominal_np;
    action.q_y = std::sqrt(config.weight_min[0] * config.weight_max[0]);
    action.q_psi = std::sqrt(config.weight_min[1] * config.weight_max[1]);
    action.q_vy = std::sqrt(config.weight_min[2] * config.weight_max[2]);
    action.q_r = std::sqrt(config.weight_min[3] * config.weight_max[3]);
    action.r_steer = std::sqrt(config.weight_min[4] * config.weight_max[4]);
    action.r_delta = std::sqrt(config.weight_min[5] * config.weight_max[5]);
    return action;
}

RlObservation makeRlObservation(const VehicleState& state,
                                const DoubleLaneChange& reference,
                                int current_np, double distance_m,
                                double speed_scale, const RlConfig& config,
                                const MpcTuningAction& current_action,
                                double road_grip, double previous_steer,
                                bool grip_mode, double previous_plan, double steer_scale) {
    if (!(std::isfinite(road_grip) && road_grip > 0.0))
        throw std::invalid_argument("RL observation requires positive road grip");
    const auto point = reference.at(state.x);
    const double preview_m = std::min(std::max(0.0, distance_m - state.x),
        std::max(5.0, state.vx * config.curvature_preview_s));
    double peak_signed_curvature = point.curvature;
    double curvature_change = 0.0;
    for (int i = 1; i <= 12; ++i) {
        const double future = reference.at(state.x + preview_m * i / 12.0).curvature;
        if (std::abs(future) > std::abs(peak_signed_curvature))
            peak_signed_curvature = future;
        curvature_change = std::max(curvature_change,
                                     std::abs(future - point.curvature));
    }
    const auto scale_state = [](double value, double minimum, double maximum) {
        return clip(2.0 * (std::log(value) - std::log(minimum)) /
            (std::log(maximum) - std::log(minimum)) - 1.0, -1.0, 1.0);
    };
    if (!(steer_scale > 0.0)) throw std::invalid_argument("RL steering observation scale must be positive");
    return {clip(state.vx / speed_scale, 0.0, 2.0),
            clip(std::asinh((state.y - point.y) / config.reward.ey_scale_m), -4.0, 4.0),
            clip(std::asinh(angleDifference(state.psi, point.psi) /
                            config.reward.epsi_scale_rad), -4.0, 4.0),
            clip(state.vy / 5.0, -5.0, 5.0),
            clip(state.r, -5.0, 5.0),
            clip(point.curvature * 30.0, -5.0, 5.0),
            clip(peak_signed_curvature * 30.0, -5.0, 5.0),
            clip(curvature_change * 30.0, 0.0, 5.0),
            clip(2.0 * (current_np - config.min_np) /
                 static_cast<double>(config.max_np - config.min_np) - 1.0, -1.0, 1.0),
            clip(state.vx * state.vx * std::abs(peak_signed_curvature) /
                 (9.81 * road_grip),
                 0.0, config.grip_demand_clip),
            scale_state(current_action.q_y, config.weight_min[0], config.weight_max[0]),
            scale_state(current_action.q_psi, config.weight_min[1], config.weight_max[1]),
            config.augmented_observation ? clip(previous_steer / steer_scale, -1.0, 1.0) : scale_state(current_action.q_vy, config.weight_min[2], config.weight_max[2]),
            config.augmented_observation ? (grip_mode ? 1.0 : 0.0) : scale_state(current_action.q_r, config.weight_min[3], config.weight_max[3]),
            config.augmented_observation ? clip(previous_plan / steer_scale, -1.0, 1.0) : scale_state(current_action.r_steer, config.weight_min[4], config.weight_max[4]),
            scale_state(current_action.r_delta, config.weight_min[5], config.weight_max[5]),
            config.train_mu_min == config.train_mu_max ? 0.0 :
                clip(2.0 * (road_grip - config.train_mu_min) /
                     (config.train_mu_max - config.train_mu_min) - 1.0, -1.0, 1.0)};
}

MpcParameters applyRlAction(const MpcParameters& base, const MpcTuningAction& action,
                            const RlConfig& config) {
    const std::array<double, 6> weights{{action.q_y, action.q_psi,
        action.q_vy, action.q_r, action.r_steer, action.r_delta}};
    for (std::size_t k : {0u, 1u, 5u})
        if (!std::isfinite(weights[k]) || weights[k] < config.weight_min[k] - 1e-9 ||
            weights[k] > config.weight_max[k] + 1e-9)
            throw std::invalid_argument("RL absolute Q/R weight is outside configured bounds");
    MpcParameters result = base;
    result.np = std::clamp(action.np, std::max(config.min_np, base.nc), config.max_np);
    result.q_y = action.q_y;
    result.q_psi = action.q_psi;
    result.r_delta = action.r_delta;
    return result;
}
