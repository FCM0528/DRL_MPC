#pragma once
#include "reference.h"
#include "vehicle.h"
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
    // R penalizes deviation from feedforward steering; R_delta penalizes changes.
    double r_steer = 10.0;
    double r_delta = 350.0;
    double max_steer = 0.42;           // front-wheel angle [rad]
    double max_steer_rate = 0.65;      // front-wheel angle slew [rad/s]
    double admm_rho = 20000.0;
    int admm_max_iterations = 2500;
    double admm_primal_tol = 1.0e-5;
    double admm_dual_tol = 2.0e-3;
};

// MPC result structure
struct MpcResult {
    double steer = 0.0;                // [rad]
    int iterations = 0;
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    bool converged = false;
    bool grip_limited = false;         // nonlinear reference tracking, no QP solve
};

class LateralMpc {
public:
    LateralMpc(const VehicleParameters& vehicle,
               const MpcParameters& settings,
               const DoubleLaneChange& reference,
               const TireTable& tires);
    MpcResult control(const VehicleState& plant, double previous_steer);
    MpcResult enterGripLimitedMode(const VehicleState& plant, double previous_steer);
    bool gripLimitedModeActive() const { return grip_mode_; }
    void setGripLimitedMode(bool active) { grip_mode_ = active; }

    // Diagnostic interface: Nc optimized moves. After Nc, last move is held.
    const std::vector<double>& lastControlMoves() const { return last_solution_; }
    double predictedSteeringAt(int prediction_step) const;

private:
    VehicleParameters vehicle_;
    MpcParameters p_;
    const DoubleLaneChange& ref_;
    const TireTable& tires_;
    DiscreteErrorModel model_;
    double front_stiffness_ = 0.0;
    double rear_stiffness_ = 0.0;
    double max_lateral_accel_ = 0.0; // tire-table peak, m/s^2
    bool grip_mode_ = false;
    VehicleState operating_point_{};
    double operating_steer_ = 0.0;
    double operating_ax_ = 0.0;
    std::vector<std::vector<State6>> sensitivity_;  // [prediction stage][control move][state]
    std::vector<std::vector<double>> hessian_;       // Nc x Nc
    std::vector<std::vector<double>> cholesky_;
    std::vector<double> last_solution_;              // Nc
    std::vector<double> low_;                        // 2*Nc
    std::vector<double> high_;

    void precompute();
    MpcResult gripLimitedControl(const VehicleState& plant, double previous_steer) const;
    void buildBounds(double previous_steer);
    std::vector<double> constraintProduct(const std::vector<double>& u) const;
    std::vector<double> constraintTranspose(const std::vector<double>& v) const;
    std::vector<double> solveCholesky(const std::vector<double>& rhs) const;
    static std::vector<std::vector<double>> choleskyFactor(
        const std::vector<std::vector<double>>& matrix);
};
