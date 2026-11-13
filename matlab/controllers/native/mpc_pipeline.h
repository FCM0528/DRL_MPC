#pragma once
#include "mpc.h"
class PipelineMpc {
public:
    PipelineMpc(const VehicleParameters& vehicle,
               const MpcParameters& settings,
               const DoubleLaneChange& reference,
               const TireTable& tires);
    MpcResult control(const VehicleState& plant, double previous_steer,
                      double previous_increment = 0.0);
    void setRoadGripProfile(std::function<double(double)> grip_at_x);
    void setPredictionHorizon(int np);
    // Preserve the latest nominal trajectory when adaptive tuning changes.
    void adoptPreviousSolution(const PipelineMpc& previous);

    // Diagnostic interface: Nc absolute steering values reconstructed from
    // optimized increments. After Nc, the last steering value is held.
    const std::vector<double>& lastControlMoves() const { return last_solution_; }
    bool hasLastSolution() const { return has_solution_; }
    double predictedSteeringAt(int prediction_step) const;


    void stageOperatingPoint(const VehicleState&, double, double);
    void stageLinearization();
    void stageReference();
    void stageCost();
    void stageConstraints();
    void stageQp();
    MpcResult stageSteering();
    double stageMetric() const { return double(nominal_states_.size()); }

private:
    VehicleState stage_plant_;
    double stage_previous_ = 0.0, stage_increment_ = 0.0;
    std::vector<ReferencePoint> stage_path_, stage_ahead_, stage_behind_;
    std::vector<double> stage_curvature_, stage_gradient_, stage_moves_;
    MpcResult stage_result_;

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
