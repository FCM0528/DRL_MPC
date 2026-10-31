#pragma once
#include "reference.h"
#include "vehicle.h"
#include <string>
#include <vector>

// 控制器参数
struct MpcParameters {
    int np = 30;                       // prediction horizon (steps)
    int nc = 30;                       // independent control horizon (1 <= Nc <= Np)
    double dt = 0.05;                  // sample period [s]
    // Diagonal Q for [e_y (m), e_psi (rad), v_y (m/s), r (rad/s)].
    double q_y = 100.0;
    double q_psi = 350.0;
    double q_vy = 0.4;
    double q_r = 8.0;
    double terminal_scale = 3.0;
    // R penalizes front-wheel angle relative to curvature feedforward;
    // R_delta penalizes the optimized steering increments.
    double r_steer = 10.0;
    double r_delta = 350.0;
    double max_steer = 0.42;           // front-wheel angle [rad]
    double max_steer_rate = 0.65;      // front-wheel angle slew [rad/s]
    bool speed_rate_enabled = false;
    double speed_rate_start = 60.0 / 3.6; // m/s
    double speed_rate_end = 120.0 / 3.6;  // m/s
    double speed_rate_end_limit = 5.0 * 3.14159265358979323846 / 180.0; // rad/s
    double admm_rho = 20000.0;
    int admm_max_iterations = 2500;
    double admm_primal_tol = 1.0e-5;
    double admm_dual_tol = 2.0e-3;
    bool grip_switch_enabled = true; // allow demand-triggered backup tracker
    bool grip_state_only = false; // trigger protection on measured state rather than instantaneous reference curvature
    double grip_preview_s = 0.8;
    bool nonlinear_tire = false;
    double tire_slope_floor = 0.15;
};

double effectiveSteerRate(const MpcParameters& settings, double vx);

// MPC result structure
struct MpcResult {
    double steer = 0.0;                // [rad]
    int iterations = 0;
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    bool converged = false;
    bool grip_limited = false;         // nonlinear reference tracking, no QP solve
    std::string prediction_failure;
};

bool optimizationFailed(const MpcResult& result);

class LateralMpc {
public:
    LateralMpc(const VehicleParameters& vehicle,
               const MpcParameters& settings,
               const DoubleLaneChange& reference,
               const TireTable& tires);
    MpcResult control(const VehicleState& plant, double previous_steer);
    void setRoadGrip(double grip);
    MpcResult enterGripLimitedMode(const VehicleState& plant, double previous_steer);
    bool gripLimitedModeActive() const { return grip_mode_; }
    void setGripLimitedMode(bool active) { grip_mode_ = active; }
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
    double front_stiffness_ = 0.0;
    double rear_stiffness_ = 0.0;
    double max_lateral_accel_ = 0.0; // tire-table peak, m/s^2
    double current_steer_rate_ = 0.0;
    bool grip_mode_ = false;
    std::vector<std::vector<State6>> sensitivity_;  // [prediction stage][control move][state]
    std::vector<State6> nominal_states_;             // [0..Np]
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
    MpcResult gripLimitedControl(const VehicleState& plant, double previous_steer) const;
    void buildBounds(double previous_steer);
    std::vector<double> constraintProduct(const std::vector<double>& u) const;
    std::vector<double> constraintTranspose(const std::vector<double>& v) const;
    std::vector<double> solveCholesky(const std::vector<double>& rhs) const;
    static std::vector<std::vector<double>> choleskyFactor(
        const std::vector<std::vector<double>>& matrix);
};
