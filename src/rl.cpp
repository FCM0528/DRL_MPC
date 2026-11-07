#include "rl.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {
double clip(double x, double lo, double hi) { return std::clamp(x, lo, hi); }

double neutralWeight(const RlConfig& config, std::size_t k) {
    const auto& choices = config.weight_choices[k];
    return choices.empty() ? std::sqrt(config.weight_min[k] * config.weight_max[k])
                           : choices[choices.size() / 2];
}

struct PathPreview {
    double peak_curvature;
    double curvature_change;
};

PathPreview previewPath(const VehicleState& state, const DoubleLaneChange& reference,
                        double distance_m, double preview_s, double curvature) {
    const double preview_m = std::min(std::max(0.0, distance_m - state.x),
                                      std::max(5.0, state.vx * preview_s));
    PathPreview preview{curvature, 0.0};
    for (int i = 1; i <= 12; ++i) {
        const double future = reference.at(state.x + preview_m * i / 12.0).curvature;
        if (std::abs(future) > std::abs(preview.peak_curvature))
            preview.peak_curvature = future;
        preview.curvature_change = std::max(preview.curvature_change, std::abs(future - curvature));
    }
    return preview;
}
}

double scalarizedRlReward(double ey, double epsi, double steer, int np,
                          const MpcParameters& mpc, const RlConfig& config,
                          double previous_steer, int previous_np,
                          const MpcTuningAction& action,
                          const MpcTuningAction& previous_action,
                          double vy, double yaw_error, double sideslip) {
    const auto& r = config.reward;
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
    const std::array<double, 11> terms{{r.w_y * lateral_loss,
        r.w_peak * peak_excess * peak_excess, r.w_psi * tracking_loss(normalized_epsi),
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
    if (r.tracking_v2) {
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
        if (!choices.empty()) {
            const auto index = std::lround((request + 1.0) * (choices.size() - 1) / 2.0);
            return choices[static_cast<std::size_t>(index)];
        }
        const double lo = std::log(config.weight_min[k]);
        const double hi = std::log(config.weight_max[k]);
        const double target = 0.5 * ((1.0 - request) * lo + (1.0 + request) * hi);
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

RlObservation makeRlObservation(const VehicleState& state,
                                const DoubleLaneChange& reference,
                                int current_np, double distance_m,
                                double speed_scale, const RlConfig& config,
                                const MpcTuningAction& current_action,
                                double road_grip, double previous_steer,
                                double previous_plan, double steer_scale) {
    if (!(std::isfinite(road_grip) && road_grip > 0.0))
        throw std::invalid_argument("RL observation requires positive road grip");
    const auto point = reference.at(state.x);
    const auto preview = previewPath(state, reference, distance_m, config.curvature_preview_s,
                                     point.curvature);
    const auto scale_state = [](double value, double minimum, double maximum) {
        // Fixed paper-style error-rate weights may be zero even though the
        // historical observation normalization bounds are positive.
        if (value <= 0.0) return -1.0;
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
            clip(preview.peak_curvature * 30.0, -5.0, 5.0),
            clip(preview.curvature_change * 30.0, 0.0, 5.0),
            clip(2.0 * (current_np - config.min_np) /
                 static_cast<double>(config.max_np - config.min_np) - 1.0, -1.0, 1.0),
            clip(state.vx * state.vx * std::abs(preview.peak_curvature) /
                 (9.81 * road_grip),
                 0.0, config.grip_demand_clip),
            scale_state(current_action.q_y, config.minWeight(0), config.maxWeight(0)),
            scale_state(current_action.q_psi, config.minWeight(1), config.maxWeight(1)),
            config.augmented_observation ? clip(previous_steer / steer_scale, -1.0, 1.0) : scale_state(current_action.q_vy, config.weight_min[2], config.weight_max[2]),
            // Reserve the former mode slot for existing augmented policies.
            config.augmented_observation ? 0.0 : scale_state(current_action.q_r, config.weight_min[3], config.weight_max[3]),
            config.augmented_observation ? clip(previous_plan / steer_scale, -1.0, 1.0) : scale_state(current_action.r_steer, config.weight_min[4], config.weight_max[4]),
            scale_state(current_action.r_delta, config.minWeight(5), config.maxWeight(5)),
            config.train_mu_min == config.train_mu_max ? 0.0 :
                clip(2.0 * (road_grip - config.train_mu_min) /
                     (config.train_mu_max - config.train_mu_min) - 1.0, -1.0, 1.0)};
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
