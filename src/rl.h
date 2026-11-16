#pragma once
#include "mpc.h"
#include "vehicle.h"
#include "reference.h"
#include <array>
#include <functional>
#include <string>
#include <vector>

// The policy sees normalized, observable vehicle/path quantities only.
using RoadPreviewObservation = std::vector<double>;

struct RlRewardConfig {
    std::string mode = "legacy"; // legacy | joint_tracking
    double w_heading = 1.0;
    double w_course = 0.0;
    double heading_scale_rad = 0.035;
    bool tracking_v2 = false;
    bool squared_lateral_reward = false; // quadratic lateral error and uncompressed loss
    double reward_scale = 10.0; // scale the scalar reward and failure penalties
    double ey_scale_m = 0.25;
    double epsi_scale_rad = 0.10;
    double w_y = 1.0;
    double w_peak = 0.20;
    double peak_threshold_m = 0.30;
    double w_lateral_growth = 0.0;
    double lateral_growth_preview_s = 0.25;
    double w_psi = 0.17;
    double w_steer = 0.03;
    double w_steer_rate = 0.08;
    double steer_rate_scale_rad_s = 1.7453292519943295; // 100 deg/s
    double sideslip_threshold_rad = 0.08726646259971647; // 5 deg
    double w_sideslip = 0.0;
    double w_horizon = 0.005;
    double w_np_change = 0.01;
    double w_vy = 0.04;
    double w_yaw_error = 0.06;
    double w_weight_change = 0.015;
    double w_weight_deviation = 0.005;
    double failure_penalty = 100.0;
    double qp_failure_penalty = 25.0;
    double incomplete_episode_penalty = 100.0;
};

struct RlConfig {
    std::string policy_action_names; // empty retains historical prefix mapping
    int policy_action_dims = 4; // Named mapping supports 1..4; empty names retain legacy 2..4 prefix
    std::string weight_action_mode = "legacy"; // legacy | continuous_weights
    std::string mode = "fixed";       // C++ defaults to MPC control
    std::string action_profile = "full"; // full | compact | np_only
    int min_np = 10;
    int max_np = 40;
    int nominal_np = 15; // independent center for the learned Np action
    bool align_neutral_to_mpc = false;
    int np_step = 1;
    int max_np_change_per_decision = 0; // 0 disables the applied-horizon slew limit
    bool smooth_tuning = false;
    double max_weight_ratio_per_decision = 1.08;
    int decision_steps = 5;
    double discount_gamma = 0.98; // discount per policy decision
    double grip_demand_clip = 3.0; // input schema changes when not 3
    // Absolute MPC cost coefficients; these do not depend on [weights].
    std::array<double, 6> weight_min{{60.0, 210.0, 0.24, 4.8, 7.0, 245.0}};
    std::array<double, 6> weight_max{{166.66666666666666, 583.3333333333334,
                                      0.6666666666666666, 13.333333333333334,
                                      14.285714285714286, 500.0}};
    std::array<std::vector<double>, 6> weight_choices;
    double minWeight(std::size_t k) const {
        return weight_choices[k].empty() ? weight_min[k] : weight_choices[k].front();
    }
    double maxWeight(std::size_t k) const {
        return weight_choices[k].empty() ? weight_max[k] : weight_choices[k].back();
    }
    double curvature_preview_s = 1.5;
    double road_preview_min_m = 0.0;
    double road_preview_max_m = 0.0; // zero means unbounded
    std::string observation_profile = "road_preview_v2";
    bool penalize_qp_failure = false; // penalize failed optimization even if grip control takes over
    RlRewardConfig reward;
};

// Bridge action: Np, Qy, Qpsi, Rdelta. Other weights stay in [mpc].
struct MpcTuningAction {
    int np = 15;
    double q_y = 100.0;
    double q_psi = 350.0;
    double q_vy = 0.4;
    double q_r = 8.0;
    double r_steer = 10.0;
    double r_delta = 350.0;
};

struct JointRewardTerms {
    double lateral = 0, heading = 0, course = 0, peak = 0, growth = 0;
    double steering = 0, parameter_switch = 0, safety = 0, other = 0;
    double total() const {
        return lateral + heading + course + peak + growth + steering +
               parameter_switch + safety + other;
    }
};
JointRewardTerms jointRewardTerms(double ey, double epsi, double steer, int np,
    const MpcParameters& mpc, const RlConfig& config, double previous_steer,
    int previous_np, const MpcTuningAction& action,
    const MpcTuningAction& previous_action, double vy, double yaw_error,
    double sideslip, double ey_dot);

MpcTuningAction neutralRlAction(const RlConfig& config);
MpcTuningAction advanceRlTuning(const std::array<double, 4>& action,
                               const MpcTuningAction& previous,
                               const RlConfig& config);
double roadPreviewDuration(const RlConfig& config, double mpc_dt);
double roadPreviewDistance(const RlConfig& config, double mpc_dt, double x_speed);
RoadPreviewObservation makeRoadPreviewObservation(
    const VehicleState& state, const DoubleLaneChange& reference,
    const std::function<double(double)>& grip_at_x, double distance_m,
    const MpcParameters& mpc, const RlConfig& config, int current_np,
    double previous_steer, const MpcTuningAction* executed = nullptr);
MpcParameters applyRlAction(const MpcParameters& base, const MpcTuningAction& action,
                            const RlConfig& config);
double scalarizedRlReward(double ey, double epsi, double steer, int np,
                          const MpcParameters& mpc, const RlConfig& config,
                          double previous_steer = 0.0, int previous_np = -1,
                          const MpcTuningAction& action = {},
                          const MpcTuningAction& previous_action = {},
                          double vy = 0.0, double yaw_error = 0.0, double sideslip = 0.0,
                          double ey_dot = 0.0);
double lateralErrorRate(const VehicleState& state, const ReferencePoint& reference);
double rlDecisionReward(double step_reward_sum, int steps, int qp_failures,
                        bool failed, bool incomplete, const RlConfig& config);
