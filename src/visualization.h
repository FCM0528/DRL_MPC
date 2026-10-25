#pragma once
#include "carsim_plant.h"
#include "mpc.h"
#include "reference.h"
#include "vehicle.h"
#include <filesystem>
#include <string>
#include <vector>

struct SimulationRecord {
    double time = 0.0;
    VehicleState s{};
    ReferencePoint reference{};
    double lateral_error = 0.0;       // global Y - Yref [m]
    double heading_error = 0.0;       // psi - psiref [rad]
    double steer = 0.0;               // front-wheel steering [rad]
    double yaw_rate_ref = 0.0;        // [rad/s]
    int qp_iterations = 0;
    double qp_primal_residual = 0.0;
    double qp_dual_residual = 0.0;
    double qp_solve_ms = 0.0;
    bool qp_converged = false;
    bool grip_limited_control = false; // reference tracker selected for insufficient tire force
    bool is_command = true;           // final sample is for plotting only
    bool fallback_control = false;     // MPC failed; steering returned toward zero at rate limit
    int prediction_horizon = 0;         // actual Np used for this command
    double dynamic_model_weight = 0.0;  // 0 kinematic, 1 dynamic
    double q_y = 100.0;
    double q_psi = 350.0;
    double q_vy = 0.4;
    double q_r = 8.0;
    double r_steer = 10.0;
    double r_delta = 350.0;
};

struct RunOutcome {
    double requested_distance_m = 0.0;
    std::string plant_name = "carsim";
    std::string controller_mode = "fixed";
    std::filesystem::path policy_file;
    std::string reference_source = "CarSim LTARG_TABLE (straight east road)";
    CarSimRunInfo carsim_info;
    double model_front_stiffness_n_per_rad = 0.0;
    double model_rear_stiffness_n_per_rad = 0.0;
    std::string browser_result_status;
    std::filesystem::path browser_result_file;
    bool completed = true;
    std::string stop_reason;
    std::string first_fallback_reason;
};

struct SimulationMetrics {
    double min_speed_mps = 0.0;
    double max_speed_mps = 0.0;
    double min_ground_speed_mps = 0.0;
    double max_ground_speed_mps = 0.0;
    double rmse_y = 0.0;
    double max_abs_y = 0.0;
    double rmse_heading_deg = 0.0;
    double max_abs_heading_deg = 0.0;
    double max_abs_steering_deg = 0.0;
    double max_steering_rate_deg_s = 0.0;
    double mean_qp_iterations = 0.0;
    double peak_qp_primal_residual = 0.0;
    double peak_qp_dual_residual = 0.0;
    double mean_qp_time_ms = 0.0;
    double max_qp_time_ms = 0.0;
    int qp_nonconverged_steps = 0;
    int grip_limited_steps = 0;
    int fallback_control_steps = 0;
};

SimulationMetrics exportResults(
    const std::filesystem::path& output,
    const std::vector<SimulationRecord>& rows,
    const VehicleParameters& vp,
    const MpcParameters& mp,
    const std::string& experiment_name,
    const RunOutcome& outcome);

// Overlay the two controllers on the original eight vehicle-control charts.
void exportComparisonVisuals(const std::filesystem::path& output,
    const std::vector<SimulationRecord>& fixed,
    const std::vector<SimulationRecord>& adaptive);
