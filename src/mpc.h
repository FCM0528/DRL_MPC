#pragma once
#include "reference.h"
#include "vehicle.h"
#include <functional>
#include <string>
#include <vector>

// 控制器参数
struct MpcParameters {
    int np = 30;                       // prediction horizon (steps)
    int nc = 30;                       // independent control horizon (1 <= Nc <= Np)
    double dt = 0.05;                  // sample period [s]
    // Diagonal Q for lateral position, heading/course, lateral rate and yaw-rate error.
    double q_y = 100.0;
    double q_psi = 350.0;
    bool track_course = false; // compare velocity direction to path tangent
    double q_vy = 0.4;
    double q_r = 8.0;
    double terminal_scale = 3.0;
    double stage_weight_lambda = 0.0; // paper-style exp(lambda * step / Np)
    // R penalizes front-wheel angle relative to curvature feedforward;
    // R_delta penalizes the optimized steering increments.
    double r_steer = 10.0;
    bool steer_feedforward = true; // false: paper's absolute steering penalty
    double r_delta = 350.0;
    double r_delta_change = 0.0;       // squared changes in steering increments
    double reference_curvature_sigma_m = 0.0;
    double max_steer = 0.42;           // front-wheel angle [rad]
    double max_steer_rate = 0.65;      // front-wheel angle slew [rad/s], +inf when disabled
    bool speed_rate_enabled = false;
    double speed_rate_start = 60.0 / 3.6; // m/s
    double speed_rate_end = 120.0 / 3.6;  // m/s
    double speed_rate_end_limit = 5.0 * 3.14159265358979323846 / 180.0; // rad/s
    double admm_rho = 20000.0;
    int admm_max_iterations = 2500;
    double admm_primal_tol = 1.0e-5;
    double admm_dual_tol = 2.0e-3;
    double tire_slope_floor = 0.15;
    double front_relaxation_m = 0.0; // effective lengths at zero slip
    double rear_relaxation_m = 0.0;
    bool usesTireRelaxation() const { return front_relaxation_m > 0.0 || rear_relaxation_m > 0.0; }
};

double effectiveSteerRate(const MpcParameters& settings, double vx);

// MPC result structure
struct MpcResult {
    double steer = 0.0;                // [rad]
    int iterations = 0;
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    bool converged = false;
    std::string prediction_failure;
};

class LateralMpc {
public:
    LateralMpc(const VehicleParameters& vehicle,
               const MpcParameters& settings,
               const DoubleLaneChange& reference,
               const TireTable& tires);
    MpcResult control(const VehicleState& plant, double previous_steer,
                      double previous_increment = 0.0);
    void setRoadGripProfile(std::function<double(double)> grip_at_x);
    // Preserve the latest nominal trajectory when adaptive tuning changes.
    void adoptPreviousSolution(const LateralMpc& previous);

    // Diagnostic interface: Nc absolute steering values reconstructed from
    // optimized increments. After Nc, the last steering value is held.
    const std::vector<double>& lastControlMoves() const { return last_solution_; }
    bool hasLastSolution() const { return has_solution_; }
    double predictedSteeringAt(int prediction_step) const;

private:
    VehicleParameters vehicle_;
    MpcParameters p_;
    const DoubleLaneChange& ref_;
    const TireTable& tires_;
    std::function<double(double)> road_grip_at_x_;
    double current_steer_rate_ = 0.0;
    std::vector<std::vector<PredictionState>> sensitivity_; // [stage][move][state]
    std::vector<PredictionState> nominal_states_;           // [0..Np]
    std::vector<double> nominal_increments_;         // Nc shifted warm start
    std::vector<double> nominal_steering_;           // Nc absolute wheel angles
    std::vector<std::vector<double>> hessian_;       // Nc x Nc
    std::vector<std::vector<double>> cholesky_;
    std::vector<double> last_solution_;              // Nc absolute wheel angles
    bool has_solution_ = false;
    std::vector<double> low_;                        // 2*Nc
    std::vector<double> high_;

    void precompute(const VehicleState& plant, double previous_steer,
                    double longitudinal_accel);
    MpcResult solve(const VehicleState& plant, double previous_steer, double previous_increment);
    void buildBounds(double previous_steer);
    std::vector<double> constraintProduct(const std::vector<double>& u) const;
    std::vector<double> constraintTranspose(const std::vector<double>& v) const;
    std::vector<double> solveCholesky(const std::vector<double>& rhs) const;
    static std::vector<std::vector<double>> choleskyFactor(
        const std::vector<std::vector<double>>& matrix);
};
