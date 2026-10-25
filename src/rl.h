#pragma once
#include "mpc.h"
#include "vehicle.h"
#include "reference.h"
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// The policy sees normalized, observable vehicle/path quantities only.
using RlObservation = std::array<double, 17>;

struct RlRewardConfig {
    double reward_scale = 10.0; // scale the scalar reward and failure penalties
    double ey_scale_m = 0.25;
    double epsi_scale_rad = 0.10;
    double w_y = 0.75;
    double w_peak = 0.20;
    double peak_threshold_m = 0.30;
    double w_psi = 0.25;
    double w_steer = 0.05;
    double w_steer_rate = 0.10;
    double w_horizon = 0.01;
    double w_np_change = 0.02;
    double w_vy = 0.08;
    double w_yaw_error = 0.10;
    double w_weight_change = 0.03;
    double w_weight_deviation = 0.01;
    double w_grip_limited = 0.03;
    double failure_penalty = 100.0;
    double incomplete_episode_penalty = 100.0;
};

struct RlConfig {
    std::string mode = "train";       // fixed | train | eval
    std::string algorithm = "ddpg";   // policy factory extension point
    int min_np = 10;
    int max_np = 40;
    int nominal_np = 15; // independent center for the learned Np action
    int decision_steps = 5;
    // Absolute MPC cost coefficients; these do not depend on [weights].
    std::array<double, 6> weight_min{{60.0, 210.0, 0.24, 4.8, 7.0, 245.0}};
    std::array<double, 6> weight_max{{166.66666666666666, 583.3333333333334,
                                      0.6666666666666666, 13.333333333333334,
                                      14.285714285714286, 500.0}};
    double terminal_scale = 3.0; // independent terminal Q multiplier
    double curvature_preview_s = 1.5;
    int episodes = 200;
    int seed = 42;
    double train_speed_min = 30.0 / 3.6;
    double train_speed_max = 120.0 / 3.6;
    double train_mu_min = 0.3;
    double train_mu_max = 0.9;
    std::filesystem::path model_file = "ddpg_policy.txt";
    RlRewardConfig reward;
};

// Seven bounded actions: Np and each nonzero Q/R diagonal weight.
struct MpcTuningAction {
    int np = 15;
    double q_y = 100.0;
    double q_psi = 350.0;
    double q_vy = 0.4;
    double q_r = 8.0;
    double r_steer = 10.0;
    double r_delta = 350.0;
};

class RlPolicy {
public:
    virtual ~RlPolicy() = default;
    virtual MpcTuningAction act(const RlObservation& observation,
                                bool explore) = 0;
    virtual void observe(const RlObservation& state, const MpcTuningAction& action,
                         double reward, const RlObservation& next, bool done) = 0;
    virtual void save(const std::filesystem::path& file) const = 0;
    virtual void load(const std::filesystem::path& file) = 0;
    virtual void setTrainingEpisode(int, int) {}
    virtual double explorationStdDev() const { return 0.0; }
};

std::unique_ptr<RlPolicy> createRlPolicy(const RlConfig& config);
MpcTuningAction neutralRlAction(const RlConfig& config);
RlObservation makeRlObservation(const VehicleState& state,
                                const DoubleLaneChange& reference,
                                int current_np, double distance_m,
                                double speed_scale, const RlConfig& config,
                                const MpcTuningAction& current_action = {},
                                double road_grip = 0.8);
MpcParameters applyRlAction(const MpcParameters& base, const MpcTuningAction& action,
                            const RlConfig& config);
double scalarizedRlReward(double ey, double epsi, double steer, int np,
                          const MpcParameters& mpc, const RlConfig& config,
                          double previous_steer = 0.0, int previous_np = -1,
                          const MpcTuningAction& action = {},
                          const MpcTuningAction& previous_action = {},
                          double vy = 0.0, double yaw_error = 0.0,
                          bool grip_limited = false);

struct AppConfig;
void runRlTraining(const AppConfig& config);
void runRlHeldoutEvaluation(const AppConfig& config);
