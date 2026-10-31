#pragma once
#include "mpc.h"
#include "vehicle.h"
#include "carsim_plant.h"
#include "rl.h"
#include <filesystem>
#include <limits>
#include <string>

// All user-tunable inputs live in configs/*.ini. The structures below
// contain safe defaults so unit tests can also construct parameters directly.
struct AppConfig {
    std::filesystem::path source_config_file;
    VehicleParameters vehicle;
    std::string plant = "carsim";
    CarSimConfig carsim;
    std::filesystem::path tire_data_file;
    MpcParameters mpc;
    RlConfig rl;
    std::filesystem::path td3_policy_file;
    std::filesystem::path ddpg_expert_file;
    std::filesystem::path td3_expert_file;
    double expert_mu = std::numeric_limits<double>::quiet_NaN();
    double expert_speed_min_kmh = std::numeric_limits<double>::quiet_NaN();
    double expert_speed_max_kmh = std::numeric_limits<double>::quiet_NaN();
    double td3_mu_min = std::numeric_limits<double>::quiet_NaN();
    double td3_mu_max = std::numeric_limits<double>::quiet_NaN();
    double evaluation_focus_mu = std::numeric_limits<double>::quiet_NaN();
    std::string experiment = "baseline";
    double distance_m = 200.0;
    std::filesystem::path output = "results/baseline";
    bool open_report = true;
};

AppConfig loadConfig(const std::filesystem::path& filename);
void overrideConfig(AppConfig& config, const std::string& qualified_key,
                    const std::string& value);
void validateConfig(const AppConfig& config);
std::string configToIni(const AppConfig& config);
std::filesystem::path locateDefaultConfig(const std::filesystem::path& executable);
