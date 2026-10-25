#pragma once
#include "vehicle.h"
#include <array>
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#define CARSIM_CALL __cdecl
#else
#define CARSIM_CALL
#endif

// Import indices are zero-based and follow the order in the CarSim Run Control.
// steering_gain maps this project's front-road-wheel radians to CarSim's
// configured import units (normally degrees for wheel-angle imports).
struct CarSimConfig {
    std::filesystem::path simfile;
    std::filesystem::path solver_dll;
    bool sync_browser_result = true;
    int steering_left_index = 0;
    int steering_right_index = -1;
    double steering_gain = 180.0 / 3.14159265358979323846;
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
    VehicleParameters vehicle;
    std::filesystem::path tire_csv;
    double front_stiffness_n_per_rad = 0.0;
    double rear_stiffness_n_per_rad = 0.0;
    std::string speed_target_kmh;
};

// Read this run's solver echo after vs_initialize. Missing or unsupported
// vehicle/tire fields are errors; never substitute nominal parameters.
CarSimMpcModel readCarSimMpcModel(const std::filesystem::path& native_dir);

CarSimRunInfo readCarSimRunInfo(const std::filesystem::path& parsfile);

// Resolve the generated Simfile's relative Parsfile and place solver outputs
// under this experiment's output directory. The original CarSim files stay put.
std::filesystem::path stageCarSimSimfile(const std::filesystem::path& source,
                                         const std::filesystem::path& output_dir);

// Mirror a finished, archived SDK run into the source Run Control's LastRun
// cache so the CarSim Browser can enable its Video/Plot buttons. Existing
// cache files are backed up into the archived run before replacement.
std::filesystem::path syncCarSimBrowserResult(const std::filesystem::path& source_simfile,
                                               const std::filesystem::path& native_dir);

class CarSimPlant final {
public:
    explicit CarSimPlant(CarSimConfig config);
    ~CarSimPlant();
    CarSimPlant(const CarSimPlant&) = delete;
    CarSimPlant& operator=(const CarSimPlant&) = delete;

    VehicleState initialState() const;
    void step(VehicleState& state, double delta_front, double dt) const;

private:
    CarSimConfig config_;
    void* library_ = nullptr;
    bool initialized_ = false;
    mutable double time_ = 0.0;
    double solver_dt_ = 0.0;
    mutable std::vector<double> imports_;
    mutable std::vector<double> exports_;
    std::array<double*, 6> state_ptrs_{};

    using ReadConfiguration = void(CARSIM_CALL*)(const char*, int*, int*,
                                              double*, double*, double*);
    using Initialize = void(CARSIM_CALL*)(double, void(CARSIM_CALL*)(double, int),
                                          void(CARSIM_CALL*)(int));
    using IntegrateIo = int(CARSIM_CALL*)(double, double*, double*);
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
