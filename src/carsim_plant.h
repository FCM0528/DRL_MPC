#pragma once
#include "vehicle.h"
#include <array>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#define CARSIM_CALL __cdecl
#else
#define CARSIM_CALL
#endif

// Import indices are zero-based and follow the order in the CarSim Run Control.
// steering_gain converts radians at the selected input to CarSim import units.
struct CarSimConfig {
    std::filesystem::path simfile;
    std::filesystem::path solver_dll;
    bool sync_browser_result = true;
    int steering_left_index = 0;
    int steering_right_index = -1;
    double steering_gain = 180.0 / 3.14159265358979323846;
    bool steering_wheel_input = false;
    std::array<std::string, 6> state_keywords{
        "XCG_TM", "YCG_TM", "YAW", "VX", "VY", "AVZ"};
};

struct CarSimRunInfo {
    std::string vehicle_dataset;
    std::string road_dataset;
    std::string procedure_dataset;
    std::string reference_dataset;
    std::string speed_target_constant;
    std::string stop_time;
};

struct CarSimMpcModel {
    struct RoadGrip {
        std::vector<std::pair<double, double>> table;
        double constant = 0.0, gain = 1.0, offset = 0.0, lateral = 0.0;
        double station_start = 0.0, station_scale = 1.0;
        double at(double station) const;
        bool variable() const { return !table.empty(); }
    } road_grip;
    VehicleParameters vehicle;
    std::filesystem::path tire_csv;
    double front_stiffness_n_per_rad = 0.0;
    double rear_stiffness_n_per_rad = 0.0;
    std::string speed_target_kmh;
};

// Read this run's solver echo after vs_initialize. Missing or unsupported
// vehicle/tire fields are errors; never substitute nominal parameters.
CarSimMpcModel readCarSimMpcModel(const std::filesystem::path& native_dir);

struct CarSimSteeringModel {
    double max_front_angle = 0.0;  // rad in the MPC's road-wheel coordinates
    // Empty for direct road-wheel input. Otherwise, road wheel -> handwheel, both rad.
    std::vector<std::pair<double, double>> front_to_handwheel;
    double inputAngle(double front_angle) const;
};

CarSimSteeringModel readCarSimSteeringModel(const std::filesystem::path& native_dir,
                                           const CarSimConfig& config);

CarSimRunInfo readCarSimRunInfo(const std::filesystem::path& parsfile);

// Resolve the generated Simfile's relative Parsfile and place solver outputs
// under this experiment's output directory. The original CarSim files stay put.
std::filesystem::path stageCarSimSimfile(const std::filesystem::path& source,
                                         const std::filesystem::path& output_dir);

// Extend only the staged private station limit for the CG-distance criterion.
void extendCarSimStationStop(const std::filesystem::path& staged_parsfile, double distance_m);

// Mirror a finished, archived SDK run into the source Run Control's LastRun
// cache so the CarSim Browser can enable its Video/Plot buttons. Existing
// cache files are backed up into the archived run before replacement.
std::filesystem::path syncCarSimBrowserResult(const std::filesystem::path& source_simfile,
                                               const std::filesystem::path& native_dir);

class CarSimPlant final {
public:
    explicit CarSimPlant(CarSimConfig config, bool read_tire_states = false,
                         bool read_steering_model = false);
    ~CarSimPlant();
    CarSimPlant(const CarSimPlant&) = delete;
    CarSimPlant& operator=(const CarSimPlant&) = delete;

    VehicleState initialState() const;
    double step(VehicleState& state, double delta_front, double dt,
                double stop_x = std::numeric_limits<double>::infinity()) const;
    const CarSimSteeringModel& steeringModel() const { return steering_; }

private:
    CarSimConfig config_;
    CarSimSteeringModel steering_;
    void* library_ = nullptr;
    bool initialized_ = false;
    mutable double time_ = 0.0;
    double solver_dt_ = 0.0;
    mutable std::vector<double> imports_;
    mutable std::vector<double> exports_;
    std::array<double*, 6> state_ptrs_{};
    std::array<double*, 4> lagged_slip_ptrs_{}; // front left/right, rear left/right
    std::array<double*, 4> tire_force_ptrs_{};

    using ReadConfiguration = void(CARSIM_CALL*)(const char*, int*, int*,
                                              double*, double*, double*);
    using Initialize = void(CARSIM_CALL*)(double, void(CARSIM_CALL*)(double, int),
                                          void(CARSIM_CALL*)(int));
    using IntegrateIo = int(CARSIM_CALL*)(double, double*, double*);
    using CopyExportVars = void(CARSIM_CALL*)(double*);
    using ErrorOccurred = int(CARSIM_CALL*)();
    using ErrorMessage = char*(CARSIM_CALL*)();
    using GetVarPtr = double*(CARSIM_CALL*)(char*);
    using TerminateRun = void(CARSIM_CALL*)(double);
    using SetErrorDialog = void(CARSIM_CALL*)(int);
    ReadConfiguration read_configuration_ = nullptr;
    Initialize initialize_ = nullptr;
    IntegrateIo integrate_io_ = nullptr;
    ErrorOccurred error_occurred_ = nullptr;
    ErrorMessage error_message_ = nullptr;
    GetVarPtr get_var_ptr_ = nullptr;
    TerminateRun terminate_run_ = nullptr;
    SetErrorDialog set_error_dialog_ = nullptr;

    std::string solverError() const;
};
