#pragma once
#include <array>
#include <filesystem>
#include <utility>
#include <vector>

// 轮胎数据表，提供侧向力与滑移角和载荷的关系
class TireTable {
public:
    explicit TireTable(const std::filesystem::path& csv_file);

    double corneringStiffness(double normal_load_n) const;
    double peakLateralForce(double normal_load_n) const;
    double minLoad() const { return loads_.front(); }
    double maxLoad() const { return loads_.back(); }

private:
    std::vector<double> loads_;
    std::vector<double> slip_deg_;
    std::vector<std::vector<double>> forces_;

    double positiveForce(double slip_deg, double normal_load_n) const;
};

// 车辆物理参数
struct VehicleParameters {
    double mass = 1500.0;       // kg
    double iz = 3000.0;         // kg m^2
    double a = 1.2;             // m: CG to front axle
    double b = 1.6;             // m: CG to rear axle
    double vx = 15.0;           // m/s: nominal cruise speed
    double road_grip_scale = 1.0;      // assumed road/tire grip relative to measured table
};
// 车辆状态
struct VehicleState {
    double x = 0.0;             // global longitudinal position [m]
    double y = 0.0;             // global lateral position [m]
    double psi = 0.0;           // vehicle heading [rad]
    double vx = 15.0;           // body-fixed longitudinal speed [m/s]
    double ax = 0.0;            // measured body-vx rate from consecutive CarSim states [m/s^2]
    double vy = 0.0;            // body-fixed lateral velocity [m/s]
    double r = 0.0;             // yaw rate [rad/s]
};

// Paper's six states, reordered to CarSim coordinates: [vy,vx,psi,r,Y,X].
// The three dynamic DOF are body lateral, body longitudinal and yaw motion.
using State6 = std::array<double, 6>;
using Matrix6 = std::array<std::array<double, 6>, 6>;

std::pair<double, double> staticWheelLoads(const VehicleParameters& p);

struct DiscreteErrorModel {
    Matrix6 a{};
    State6 b{};   // front-wheel steering angle
    State6 c{};   // affine term around the measured CarSim operating point
};

DiscreteErrorModel discretizeErrorModel(const VehicleParameters& p,
                                        const TireTable& tires,
                                        const VehicleState& operating_point,
                                        double steering_angle,
                                        double longitudinal_accel, double dt);
// Geometry-based low-speed predictor with measured CarSim sideslip/yaw residuals.
// The blend is 0 below 32 km/h, 1 above 40 km/h, centered at 36 km/h.
double dynamicPredictionWeight(double speed_mps);
DiscreteErrorModel discretizeKinematicModel(const VehicleParameters& p,
                                             const VehicleState& operating_point,
                                             double steering_angle,
                                             double longitudinal_accel, double dt);
DiscreteErrorModel discretizePredictionModel(const VehicleParameters& p,
                                              const TireTable& tires,
                                              const VehicleState& operating_point,
                                              double steering_angle,
                                              double longitudinal_accel, double dt);
State6 matrixTimesState(const Matrix6& a, const State6& x);
double angleDifference(double a, double b);
