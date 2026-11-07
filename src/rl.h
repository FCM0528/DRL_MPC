#pragma once
#include "mpc.h"
#include "vehicle.h"
#include "reference.h"
#include <array>
#include <string>
#include <vector>

// The policy sees normalized, observable vehicle/path quantities only.
using RlObservation = std::array<double, 17>;

struct RlRewardConfig {
    bool tracking_v2 = false;
    bool squared_lateral_reward = false; // quadratic lateral error and uncompressed loss
    double reward_scale = 10.0; // scale the scalar reward and failure penalties
    double ey_scale_m = 0.25;
    double epsi_scale_rad = 0.10;
    double w_y = 1.0;
    double w_peak = 0.20;
    double peak_threshold_m = 0.30;
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
    std::string mode = "fixed";       // C++ defaults to MPC control
    std::string action_profile = "full"; // full | compact | np_only
    int min_np = 10;
    int max_np = 40;
    int nominal_np = 15; // independent center for the learned Np action
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
    bool augmented_observation = false; // compact policy additionally sees steering and protection state
    bool penalize_qp_failure = false; // penalize failed optimization even if grip control takes over
    double train_mu_min = 0.3;
    double train_mu_max = 0.9;
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

MpcTuningAction neutralRlAction(const RlConfig& config);
MpcTuningAction advanceRlTuning(const std::array<double, 4>& action,
                               const MpcTuningAction& previous,
                               const RlConfig& config);
RlObservation makeRlObservation(const VehicleState& state,
                                const DoubleLaneChange& reference,
                                int current_np, double distance_m,
                                double speed_scale, const RlConfig& config,
                                const MpcTuningAction& current_action = {},
                                double road_grip = 0.8,
                                double previous_steer = 0.0,
                                double previous_plan = 0.0,
                                double steer_scale = 0.42);
MpcParameters applyRlAction(const MpcParameters& base, const MpcTuningAction& action,
                            const RlConfig& config);
double scalarizedRlReward(double ey, double epsi, double steer, int np,
                          const MpcParameters& mpc, const RlConfig& config,
                          double previous_steer = 0.0, int previous_np = -1,
                          const MpcTuningAction& action = {},
                          const MpcTuningAction& previous_action = {},
                          double vy = 0.0, double yaw_error = 0.0, double sideslip = 0.0);
double rlDecisionReward(double step_reward_sum, int steps, int qp_failures,
                        bool failed, bool incomplete, const RlConfig& config);
