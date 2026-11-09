#pragma once
#include "mpc.h"
#include "vehicle.h"
#include "carsim_plant.h"
#include "rl.h"
#include <filesystem>
#include <string>

struct HorizonSchedule {
    int short_np = 0; // 0 disables scheduling
    double lookahead_s = 0.8;
    double enter_ratio = 1.15;
    double exit_ratio = 0.8;
};

// All user-tunable inputs live in configs/*.ini. The structures below
// contain safe defaults so unit tests can also construct parameters directly.
struct AppConfig {
    std::filesystem::path source_config_file;
    VehicleParameters vehicle;
    std::string plant = "carsim";
    CarSimConfig carsim;
    std::filesystem::path tire_data_file;
    MpcParameters mpc;
    HorizonSchedule horizon_schedule;
    bool steer_limit_from_carsim = false;
    RlConfig rl;
    std::string experiment = "baseline";
    double distance_m = 120.0;
    bool stop_at_reference_end = false;
    std::filesystem::path output = "results/baseline";
    bool open_report = true;
};

AppConfig loadConfig(const std::filesystem::path& filename);
void overrideConfig(AppConfig& config, const std::string& qualified_key,
                    const std::string& value);
void validateConfig(const AppConfig& config);
void resolveReferenceEnd(AppConfig& config, const DoubleLaneChange& reference);
std::string configToIni(const AppConfig& config);
std::filesystem::path locateDefaultConfig(const std::filesystem::path& executable);
