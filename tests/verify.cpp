#include "config.h"
#include "carsim_plant.h"
#include "mpc.h"
#include "reference.h"
#include "vehicle.h"
#include "rl.h"
#include "visualization.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
DoubleLaneChange testReference() {
    const auto path = std::filesystem::temp_directory_path() / "mpc_carsim_reference_fixture.par";
    {
        std::ofstream out(path);
        out << "set_description path_id Straight East\nLTARG_TABLE SPLINE_FLAT\n"
               "0, 0\n40, 0\n60, 3\n80, 3\n100, -1\n150, -1\nENDTABLE\n";
    }
    auto ref = DoubleLaneChange::fromCarSimParsfile(path);
    std::filesystem::remove(path);
    require(std::abs(ref.at(150).y + 1.0) < 1e-12, "CarSim path endpoint incorrect");
    require(ref.at(-1).x_min == 0.0 && ref.at(151).x_max == 150.0,
            "Reference table plot bounds were not preserved");
    AppConfig config;
    overrideConfig(config, "simulation.distance_m", "reference_end");
    resolveReferenceEnd(config, ref);
    require(config.distance_m == 150.0, "Automatic stop did not use the reference endpoint");
    overrideConfig(config, "simulation.distance_m", "100");
    resolveReferenceEnd(config, ref);
    require(config.distance_m == 100.0, "Explicit stop did not override the reference endpoint");
    {
        std::ofstream out(path);
        out << "set_description path_id Straight East\nLTARG_TABLE SPLINE_FLAT\n";
        for (int x = 0; x <= 10; ++x) out << x << ", 0\n";
        out << "11, 0.002\n12, 0.0024\n13, 0.0029\n14, 0.0035\n20, 0.0111\nENDTABLE\n";
    }
    const auto quantized = DoubleLaneChange::fromCarSimParsfile(path);
    std::filesystem::remove(path);
    const auto native = quantized.at(10.0);
    require(quantized.controlCurvature(10.0, 0.0) == native.curvature &&
            std::abs(quantized.controlCurvature(10.0, 3.0)) < 0.25 * std::abs(native.curvature) &&
            quantized.at(11.0).y == 0.002,
            "Curvature preview must suppress entry quantization and preserve the native target");
    return ref;
}
void testTireZeroLoadColumn() {
    const auto path = std::filesystem::temp_directory_path() /
        "mpc_tire_zero_load_fixture.csv";
    {
        std::ofstream out(path);
        out << "slip_deg,0,100,200\n0,0,0,0\n1,0,10,20\n2,0,15,30\n";
    }
    TireTable tire(path);
    std::filesystem::remove(path);
    constexpr double rad = 3.14159265358979323846 / 180.0;
    require(std::abs(tire.lateralForce(rad, 95.0) - 9.5) < 1e-10 &&
            tire.lateralForce(rad, 0.0) == 0.0,
            "CarSim zero-load column should interpolate within the tire table");
    bool rejected = false;
    try { (void)tire.lateralForce(rad, 210.0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Tire loads beyond the CarSim table must be rejected");
    rejected = false;
    try { (void)tire.lateralForce(3.0 * rad, 95.0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Tire slip beyond the CarSim table must be rejected");
}
void testConfig() {
    auto config = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/default.ini");
    require(config.mpc.track_course && config.mpc.nc <= config.mpc.np &&
            config.mpc.front_relaxation_m == 0.40 && config.mpc.rear_relaxation_m == 0.30 &&
            config.mpc.q_y > 0.0 && config.mpc.q_psi == 2800.0 &&
            config.mpc.q_vy == 650.0 && config.mpc.q_r == 150.0 &&
            config.mpc.r_steer == 33000.0 && config.mpc.steer_feedforward &&
            config.mpc.r_delta > 0.0 && config.mpc.r_delta_change == 240000.0 &&
            config.mpc.terminal_scale == 1.0 && config.mpc.stage_weight_lambda == 0.0,
            "Default incremental MPC did not load configured base weights");
    require(config.plant == "carsim" && !config.carsim.sync_browser_result,
            "CarSim default configuration incorrect");
    require(config.stop_at_reference_end && config.steer_limit_from_carsim && !config.mpc.speed_rate_enabled &&
            config.mpc.reference_curvature_sigma_m == 2.5 &&
            std::isinf(effectiveSteerRate(config.mpc, 95.0 / 3.6)),
            "Fixed MPC must use the CarSim wheel range without an actuator slew limit");
    const auto rl_config = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/carsim_python_rl.ini");
    const auto& a = config.mpc;
    const auto& b = rl_config.mpc;
    require(rl_config.steer_limit_from_carsim && b.reference_curvature_sigma_m == a.reference_curvature_sigma_m &&
            b.track_course && b.usesTireRelaxation() && b.np == rl_config.rl.nominal_np &&
            b.nc <= rl_config.rl.min_np && b.tire_slope_floor == a.tire_slope_floor &&
            std::isinf(b.max_steer_rate) &&
            b.q_vy == a.q_vy && b.q_r == a.q_r && b.r_steer == a.r_steer &&
            b.steer_feedforward && b.r_delta_change == a.r_delta_change,
            "Discrete Python policy must use the calibrated fixed MPC model");
    require(a.dt == b.dt && a.terminal_scale == b.terminal_scale &&
            a.stage_weight_lambda == b.stage_weight_lambda &&
            a.admm_rho == b.admm_rho && a.admm_max_iterations == b.admm_max_iterations &&
            a.admm_primal_tol == b.admm_primal_tol && a.admm_dual_tol == b.admm_dual_tol,
            "Fixed and Python MPC shared settings differ");
    auto schedule = config.mpc;
    schedule.speed_rate_enabled = true;
    schedule.max_steer_rate = 1.0;
    schedule.speed_rate_start = 60.0 / 3.6;
    schedule.speed_rate_end = 120.0 / 3.6;
    schedule.speed_rate_end_limit = 5.0 * 3.14159265358979323846 / 180.0;
    require(std::abs(effectiveSteerRate(schedule, 60.0 / 3.6) - schedule.max_steer_rate) < 1e-12 &&
            std::abs(effectiveSteerRate(schedule, 120.0 / 3.6) - schedule.speed_rate_end_limit) < 1e-12 &&
            effectiveSteerRate(schedule, 90.0 / 3.6) < schedule.max_steer_rate &&
            effectiveSteerRate(schedule, 90.0 / 3.6) > schedule.speed_rate_end_limit,
            "Speed-dependent steering rate interpolation incorrect");
    schedule.speed_rate_enabled = false;
    require(effectiveSteerRate(schedule, 120.0 / 3.6) == schedule.max_steer_rate,
            "Disabled speed-dependent limit changed physical steering rate");
    overrideConfig(config, "mpc.np", "30");
    overrideConfig(config, "mpc.nc", "10");
    overrideConfig(config, "weights.q_y", "250");
    overrideConfig(config, "rl.max_np_change_per_decision", "2");
    overrideConfig(config, "rl.smooth_tuning", "true");
    overrideConfig(config, "rl.train_speed_min_kmh", "36"); // archived setting
    overrideConfig(config, "rl.tuning_ay_start_mps2", "0.35"); // archived setting
    validateConfig(config);
    require(config.mpc.np == 30 && config.mpc.nc == 10 && config.mpc.q_y == 250 &&
            config.rl.max_np_change_per_decision == 2 && config.rl.smooth_tuning,
            "MPC or RL override incorrect");
    const auto saved = configToIni(config);
    require(saved.find("[carsim]") != std::string::npos &&
            saved.find("max_np_change_per_decision = 2") != std::string::npos &&
            saved.find("smooth_tuning = true") != std::string::npos &&
            saved.find("train_speed_min_kmh") == std::string::npos &&
            saved.find("tuning_ay_start_mps2") == std::string::npos &&
            saved.find("max_speed_drop") == std::string::npos &&
            saved.find("[reference]") == std::string::npos,
            "Effective config retained legacy plant settings");
    const auto fixture = std::filesystem::temp_directory_path() /
        ("mpc_config_inherit_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(fixture);
    const auto write = [](const std::filesystem::path& path, const std::string& content) {
        std::ofstream out(path, std::ios::binary);
        out << content;
        if (!out) throw std::runtime_error("Cannot write config fixture");
    };
    config.mpc.np = 21;
    config.mpc.q_y = 432;
    write(fixture / "base.ini", configToIni(config));
    write(fixture / "overlay.ini", "[inherit]\nbase = base.ini\n[mpc]\nnonlinear_tire = true\n");
    auto inherited = loadConfig(fixture / "overlay.ini");
    require(inherited.mpc.np == 21 && inherited.mpc.q_y == 432,
            "Inherited values or overlay incorrect");
    config.mpc.np = 22;
    config.mpc.q_y = 321;
    write(fixture / "base.ini", configToIni(config));
    inherited = loadConfig(fixture / "overlay.ini");
    require(inherited.mpc.np == 22 && inherited.mpc.q_y == 321,
            "Changed base configuration was not reloaded");
    bool linear_rejected = false;
    try { overrideConfig(config, "mpc.nonlinear_tire", "false"); }
    catch (const std::invalid_argument&) { linear_rejected = true; }
    require(linear_rejected, "Removed linear-tire mode was accepted");
    write(fixture / "cycle.ini", "[inherit]\nbase = cycle.ini\n");
    bool cycle_rejected = false;
    try { loadConfig(fixture / "cycle.ini"); } catch (const std::runtime_error&) { cycle_rejected = true; }
    require(cycle_rejected, "Circular configuration inheritance accepted");
    std::filesystem::remove_all(fixture);
    bool rejected = false;
    try { overrideConfig(config, "simulation.plant", "bicycle"); validateConfig(config); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Legacy bicycle mode accepted");
    std::cout << "PASS: CarSim configuration\n";
}void testCarSimBrowserResultSync() {
    const auto temp = std::filesystem::temp_directory_path();
    const auto root = temp / ("mpc_carsim_browser_sync_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto gui = root / "Results" / "Run_fixture";
    const auto archive = root / "archive";
    std::filesystem::create_directories(gui);
    std::filesystem::create_directories(archive);
    const auto write = [](const std::filesystem::path& path, const std::string& value) {
        std::ofstream out(path, std::ios::binary);
        out << value;
        if (!out) throw std::runtime_error("Cannot write Browser sync test fixture");
    };
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    const auto simfile = root / "simfile.sim";
    write(simfile, "SIMFILE\nSET_MACRO $(ROOT)$ Run_fixture\nDATADIR " + root.string() +
          "\nINPUT Results\\$(ROOT)$\\Run_all.par\n"
          "FILEBASE Results\\$(ROOT)$\\LastRun\nEND\n");
    write(gui / "Run_all.par", "PARSFILE\nSPEED_TARGET_CONSTANT 30\nEND\n");
    write(archive / "Run_visualizer.par", read(gui / "Run_all.par"));
    write(archive / "LastRun.vs", "new header");
    write(archive / "LastRun.vsb", "new data");
    write(archive / "LastRun_echo.par", "new echo");
    write(archive / "LastRun_end.par", "new end");
    write(archive / "LastRun_log.txt", "new log");
    write(gui / "LastRun.vs", "old header");
    write(gui / "LastRun.vsb", "old data");
    const auto published = syncCarSimBrowserResult(simfile, archive);
    require(published == gui / "LastRun.vs" && read(published) == "new header" &&
            read(gui / "LastRun.vsb") == "new data", "Browser cache sync failed");
    require(read(archive / "browser_previous_cache" / "LastRun.vs") == "old header" &&
            read(archive / "browser_previous_cache" / "LastRun.vsb") == "old data",
            "Previous Browser output was not archived");
    write(gui / "Run_all.par", "PARSFILE\nSPEED_TARGET_CONSTANT 60\nEND\n");
    bool rejected_stale_run = false;
    try { syncCarSimBrowserResult(simfile, archive); }
    catch (const std::runtime_error&) { rejected_stale_run = true; }
    require(rejected_stale_run && read(gui / "LastRun.vs") == "new header",
            "Changed CarSim source must not publish stale output");
    const std::string original_path =
        "PARSFILE\nset_description path_id Straight East\nSPEED_TARGET_CONSTANT 40\n"
        "LTARG_TABLE SPLINE_FLAT\n0, 0\n1, 1\n2, 0\nENDTABLE\nEXPORT Yaw\nEND\n";
    write(gui / "Run_all.par", original_path);
    const auto staged = root / "staged_run";
    stageCarSimSimfile(simfile, staged);
    const auto solver = DoubleLaneChange::fromCarSimParsfile(staged / "Run_all.par");
    const auto visualizer = DoubleLaneChange::fromCarSimParsfile(staged / "Run_visualizer.par");
    require(solver.at(1).y == 1.0 && visualizer.at(1).y == 1.0 &&
            read(staged / "Run_all.par").find("SPEED_TARGET_CONSTANT 40") != std::string::npos &&
            read(staged / "Run_all.par").find("EXPORT Yaw") == std::string::npos &&
            read(staged / "Run_all.par").find("EXPORT Steer_L1\nEXPORT Steer_R1\nPORTS_EXP 1,2") != std::string::npos &&
            read(staged / "Run_visualizer.par") == original_path &&
            read(gui / "Run_all.par") == original_path,
            "Both run copies must preserve the native reference");
    require(std::filesystem::equivalent(root.parent_path(), temp),
            "Unexpected Browser sync test directory");
    std::filesystem::remove_all(root);
    std::cout << "PASS: CarSim Browser sync archives cache and rejects changed source\n";
}
void testCarSimRackSteeringLimit() {
    const auto dir = std::filesystem::temp_directory_path() / "mpc_carsim_rack_fixture";
    std::filesystem::create_directories(dir);
    {
        std::ofstream echo(dir / "LastRun_echo.par");
        echo << "OPT_STEER_DEF 1\nOPT_RACK(1) 1\n"
                "RACK_KIN_TABLE(1,1) SPLINE\n-75, -30\n0, 0\n75, 42\nENDTABLE\n"
                "RACK_KIN_TABLE(1,2) SPLINE\n-75, -42\n0, 0\n75, 30\nENDTABLE\n"
                "RACK_KIN_GAIN(1,2) 0.8\nRACK_KIN_OFFSET(1,2) 1\n"
                "IMPORT IMP_STEER_L1 REPLACE 0 ; deg ! #1\n"
                "IMPORT IMP_STEER_R1 REPLACE 0 ; deg ! #2\n";
    }
    CarSimConfig config;
    config.steering_right_index = 1;
    require(std::abs(readCarSimSteeringModel(dir, config).max_front_angle * config.steering_gain - 25.0) < 1e-9,
            "CarSim rack range did not account for both wheels and table modifiers");
    std::swap(config.steering_left_index, config.steering_right_index);
    bool rejected = false;
    try { (void)readCarSimSteeringModel(dir, config); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Automatic CarSim steering limit accepted mismatched imports");
    config.steering_left_index = 0;
    config.steering_right_index = -1;
    config.steering_wheel_input = true;
    const auto writeHandwheel = [&](double rack_pitch) {
        std::ofstream echo(dir / "LastRun_echo.par");
        echo << "OPT_STEER_DEF 1\nOPT_RACK(1) 1\nOPT_STR_BY_TRQ 0\nOPT_STEER_EXT(1) 0\n"
                "RACK_KIN_TABLE(1,1) SPLINE\n-60, -30\n0, 0\n60, 30\nENDTABLE\n"
                "RACK_KIN_TABLE(1,2) SPLINE\n-50, -25\n0, 0\n50, 25\nENDTABLE\n"
                "SET_UNITS_TABLE_ROW RACK_TRAVEL_TABLE(1) rev\n"
                "RACK_TRAVEL_COEFFICIENT(1) " << rack_pitch << "\n"
                "IMPORT IMP_STEER_SW REPLACE 0 ; deg ! #1\n";
    };
    writeHandwheel(40);
    const auto slower = readCarSimSteeringModel(dir, config);
    writeHandwheel(50);
    const auto faster = readCarSimSteeringModel(dir, config);
    const double front = 10.0 / config.steering_gain;
    require(std::abs(slower.inputAngle(front) * config.steering_gain - 180.0) < 1e-9 &&
            std::abs(faster.inputAngle(front) * config.steering_gain - 144.0) < 1e-9 &&
            std::abs(faster.max_front_angle * config.steering_gain - 25.0) < 1e-9,
            "Handwheel conversion did not follow the CarSim rack pitch or shared rack travel");
    {
        std::ofstream echo(dir / "LastRun_echo.par", std::ios::app);
        echo << "RACK_TRAVEL_TABLE(1) LINEAR\n-2, -60\n0, 0\n1, 50\nENDTABLE\n";
    }
    const auto nonlinear = readCarSimSteeringModel(dir, config);
    require(std::abs(nonlinear.inputAngle(-front) * config.steering_gain + 240.0) < 1e-9,
            "Handwheel conversion did not invert the nonlinear rack travel table");
    std::filesystem::remove(dir / "LastRun_echo.par");
    std::filesystem::remove(dir);
}

void testCarSimMpcModel() {
    const auto root = std::filesystem::temp_directory_path() /
        ("mpc_carsim_model_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    const auto echo = root / "LastRun_echo.par";
    {
        std::ofstream out(echo, std::ios::binary);
        out << "! M_TL 1501 ; kg\n! IZZ_TL 2192.089539 ; kg-m2\n"
               "! LX_CG_TL 1049.562958 ; mm\nLX_AXLE(1) 0 ; mm\n"
               "LX_AXLE(2) 2910 ; mm\nMU_ROAD_CONSTANT(1) 0.8\n"
               "MU_ROAD_GAIN(1) 1\nMU_ROAD_OFFSET(1) 0\n"
               "MU_ROAD_COMBINE(1) ADD\nMU_ROAD_L_CONSTANT(1) 0\n"
               "SPEED_TARGET_CONSTANT(1) 45 ; km/h\n";
        for (int axle = 1; axle <= 2; ++axle)
            for (int side = 1; side <= 2; ++side) {
                const auto suffix = "(" + std::to_string(axle) + "," +
                                    std::to_string(side) + ")";
                out << "FY_TIRE_CARPET" << suffix << " 2D_FROM_ZERO\n"
                       "0, 1000, 5000, 9000\n0.5, 100, 500, 900\n"
                       "1, 200, 1000, 1800\nENDTABLE\n";
                out << "FY_TIRE_GAIN" << suffix << " 1\n"
                    << "FY_TIRE_OFFSET" << suffix << " 0\n"
                    << "ALPHA_START_FY_TIRE" << suffix << " 0\n"
                    << "ALPHA_SCALE_FY_TIRE" << suffix << " 1\n"
                    << "FZ_START_FY_TIRE" << suffix << " 0\n"
                    << "FZ_SCALE_FY_TIRE" << suffix << " 1\n"
                    << "MU_REF_Y" << suffix << " 1\n";
            }
    }
    const auto model = readCarSimMpcModel(root);
    TireTable exported_tire(model.tire_csv);
    require(exported_tire.minLoad() == 0.0,
            "CarSim 2D_FROM_ZERO hidden zero-load column was not reconstructed");
    require(model.vehicle.mass == 1501 &&
            std::abs(model.vehicle.iz - 2192.089539) < 1e-8 &&
            std::abs(model.vehicle.a - 1.049562958) < 1e-9 &&
            std::abs(model.vehicle.b - 1.860437042) < 1e-9 &&
            model.vehicle.road_grip_scale == 0.8 &&
            model.front_stiffness_n_per_rad > 0 &&
            model.rear_stiffness_n_per_rad > 0 &&
            model.speed_target_kmh == "45", "CarSim MPC model values are incorrect");
    {
        std::ifstream in(echo, std::ios::binary);
        std::string content{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        const auto pos = content.find("MU_ROAD_CONSTANT(1) 0.8");
        require(pos != std::string::npos, "Constant grip fixture missing");
        content.replace(pos, std::string("MU_ROAD_CONSTANT(1) 0.8").size(),
            "MU_ROAD_TABLE(1) STEP\n0, 0.9\n100, 0.6\n200, 0.8\nENDTABLE\n"
            "SSTART_MU_ROAD(1) 10\nSSCALE_MU_ROAD(1) 2\nROAD_PATH_ID(1) 1\nPATH_ID_DM 1");
        std::ofstream out(echo, std::ios::binary | std::ios::trunc); out << content;
    }
    const auto variable = readCarSimMpcModel(root);
    require(variable.road_grip.variable() && variable.road_grip.at(0) == 0.9 &&
            variable.road_grip.at(209.99) == 0.9 && variable.road_grip.at(210) == 0.6 &&
            variable.road_grip.at(410) == 0.8,
            "CarSim STEP road grip or station scaling incorrect");
    const auto stop_file=root / "private_stop.par";
    { std::ofstream out(stop_file); out << "OPT_STOP 1\nSSTOP 180\nTSTOP 20\n"; }
    extendCarSimStationStop(stop_file,180.0);
    { std::ifstream in(stop_file); const std::string content{std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
      require(content.find("SSTOP 190")!=std::string::npos && content.find("OPT_STOP 1")!=std::string::npos,
              "Staged CarSim station limit was not extended"); }
    { std::ofstream out(stop_file); out << "OPT_STOP 0\nSSTOP = SSTART + 1 ;\n"; }
    extendCarSimStationStop(stop_file,180.0);
    { std::ifstream in(stop_file); const std::string content{std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
      require(content.find("SSTOP = SSTART + 1")!=std::string::npos,
              "Inactive CarSim stop expression changed"); }
    std::filesystem::remove_all(root);
    std::cout << "PASS: CarSim solver echo supplies the MPC vehicle and tire model\n";
}
void testRlModule() {
    auto config = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/carsim_python_rl.ini");
    require(config.stop_at_reference_end && config.rl.reward.tracking_v2,
            "Training must use the CarSim reference end and current reward");
    const auto neutral = neutralRlAction(config.rl);
    require(neutral.np == config.mpc.np &&
            std::abs(neutral.q_y - config.mpc.q_y) < 1e-12 &&
            std::abs(neutral.q_psi - config.mpc.q_psi) < 1e-12 &&
            std::abs(neutral.r_delta - config.mpc.r_delta) < 1e-12,
            "Neutral action must match the fixed MPC parameters");
    const std::array<double, 4> high{1, 1, 1, 1}, low{-1, -1, -1, -1};
    const auto raised = advanceRlTuning(high, neutral, config.rl);
    const auto lowered = advanceRlTuning(low, raised, config.rl);
    require(raised.np == neutral.np + 2 && lowered.np == neutral.np &&
            raised.q_y == 13800.0 && raised.q_psi == 5600.0 && raised.r_delta == 27000.0 &&
            lowered.q_y == 4600.0 && lowered.q_psi == 1400.0 && lowered.r_delta == 6750.0,
            "Discrete actions must reach weight endpoints and preserve the even horizon slew");
    auto discrete = config.rl;
    discrete.max_np_change_per_decision = 0;
    for (int np = 10; np <= 50; np += 2) {
        const int span = np < neutral.np ? neutral.np - discrete.min_np : discrete.max_np - neutral.np;
        const double request = static_cast<double>(np - neutral.np) / span;
        require(advanceRlTuning({request, 0, 0, 0}, neutral, discrete).np == np,
                "Every even horizon from 10 to 50 must be reachable");
    }
    const auto intermediate = advanceRlTuning({0, -0.5, 0.35, 0.67}, neutral, discrete);
    require(intermediate.q_y == 7500.0 && intermediate.q_psi == 3500.0 &&
            intermediate.r_delta == 21500.0, "Interior requests must select exact array entries");
    const auto saved = configToIni(config);
    require(saved.find("np_step = 2") != std::string::npos &&
            saved.find("q_y_values = 4600, 6000, 7500, 9200, 10500, 12000, 13800") != std::string::npos &&
            saved.find("q_psi_values = ") != std::string::npos && saved.find("r_delta_values = ") != std::string::npos,
            "Effective configuration must retain the discrete action space");
    RlConfig legacy;
    legacy.smooth_tuning = true;
    const auto legacy_neutral = neutralRlAction(legacy);
    const auto legacy_raised = advanceRlTuning(high, legacy_neutral, legacy);
    require(legacy_raised.q_y > legacy_neutral.q_y &&
            legacy_raised.q_y / legacy_neutral.q_y <= legacy.max_weight_ratio_per_decision + 1e-12 &&
            legacy_raised.q_y < legacy.maxWeight(0),
            "Archived continuous actions must retain weight smoothing");
    auto base = config.mpc;
    base.q_vy = 0.37;
    base.q_r = 7.5;
    base.r_steer = 9.0;
    const auto tuned = applyRlAction(base, neutral, config.rl);
    require(tuned.np == config.rl.nominal_np && tuned.q_vy == base.q_vy &&
            tuned.q_r == base.q_r && tuned.r_steer == base.r_steer,
            "Untrained MPC weights changed by a Python action");
    const auto reference = testReference();
    VehicleState state;
    state.vx = 60.0 / 3.6;
    const auto obs = makeRlObservation(state, reference, neutral.np,
        config.distance_m, 30.0, config.rl, neutral, 0.8);
    require(obs.size() == 17 && std::all_of(obs.begin(), obs.end(),
        [](double x) { return std::isfinite(x); }), "Bridge observation schema changed");
    auto paper_action = neutral;
    paper_action.q_vy = 0.0;
    paper_action.q_r = 0.0;
    paper_action.r_steer = config.mpc.r_steer;
    const auto paper_obs = makeRlObservation(state, reference, paper_action.np,
        config.distance_m, 30.0, config.rl, paper_action, 0.8);
    require(std::all_of(paper_obs.begin(), paper_obs.end(),
        [](double x) { return std::isfinite(x); }),
        "Zero error-rate weights made the RL observation nonfinite");
    const double reward = scalarizedRlReward(0.1, 0.02, 0.01, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral);
    require(std::isfinite(reward) && reward < 0.0, "C++ reward is invalid");
    const double accurate = scalarizedRlReward(0.0, 0.0, 0.0, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral);
    require(std::abs(accurate) < 1e-12, "Zero-loss reward must be zero");
    const double moderate = scalarizedRlReward(0.05, 0.0, 0.0, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral);
    const double at_threshold = scalarizedRlReward(0.25, 0.0, 0.0, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral);
    const double large = scalarizedRlReward(0.35, 0.0, 0.0, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral);
    require(accurate > moderate && moderate > at_threshold && at_threshold > large &&
            std::abs(at_threshold / moderate - 25.0) < 1e-12 && large / moderate > 49.0,
            "Lateral reward must be quadratic with extra loss above 0.25 m");
    const auto& r = config.rl.reward;
    const double rate_loss = scalarizedRlReward(0.0, 0.0, 0.0, neutral.np,
        config.mpc, config.rl, -r.steer_rate_scale_rad_s * config.mpc.dt,
        neutral.np, neutral, neutral);
    require(std::abs(rate_loss + r.reward_scale * r.w_steer_rate) < 1e-12,
            "Rate loss must stay active without a steering slew limit");
    const double spike_loss = scalarizedRlReward(0.0, 0.0, 0.0, neutral.np,
        config.mpc, config.rl, -3.0 * r.steer_rate_scale_rad_s * config.mpc.dt,
        neutral.np, neutral, neutral);
    require(std::abs(spike_loss + 8.0 * r.reward_scale * r.w_steer_rate) < 1e-12,
            "Steering rate spikes must still incur additional loss above twice the scale");
    const double beta = 2.0 * r.sideslip_threshold_rad;
    const double sideslip_loss = scalarizedRlReward(0.0, -beta, 0.0, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral, 0.0, 0.0, beta);
    require(std::abs(sideslip_loss + r.reward_scale * r.w_sideslip) < 1e-12,
            "Matching course must leave only the excess sideslip loss");
    require(rlDecisionReward(-4.0 * r.reward_scale, 4, 1, false, false, config.rl) ==
                -1.0 - r.qp_failure_penalty &&
            rlDecisionReward(-r.reward_scale, 1, 1, false, false, config.rl) ==
                -1.0 - r.qp_failure_penalty &&
            rlDecisionReward(-r.reward_scale, 2, 0, true, true, config.rl) ==
                -1.0 - r.failure_penalty &&
            rlDecisionReward(-r.reward_scale, 1, 0, false, true, config.rl) ==
                -1.0 - r.incomplete_episode_penalty,
            "Event penalties must be independent of decision length");
    std::cout << "PASS: Python bridge observation, action and reward contracts\n";
}

void testMpcConstraints() {
    const auto ref = testReference();
    auto config = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/default.ini");
    TireTable tires(std::string(PROJECT_SOURCE_DIR) + "/configs/magic_tire.csv");
    auto model = config.vehicle;
    model.vx = 12.5;
    model.mass = 1501;
    model.iz = 2192.089539;
    model.a = 1.049562958;
    model.b = 1.860437042;
    model.road_grip_scale = 0.8;
    VehicleState operating;
    operating.vx=model.vx;
    operating.vy=0.4;
    operating.psi=0.12;
    operating.r=0.15;
    operating.x=10.0;
    operating.y=1.0;
    constexpr double steer=0.07, ax=0.2, epsilon=1e-4, small_dt=1e-5;
    const auto [front_load,rear_load]=staticWheelLoads(model);
    const double alpha_f=steer-(operating.vy+model.a*operating.r)/operating.vx;
    const double cf=2.0*model.road_grip_scale*
        (tires.lateralForce(alpha_f+1e-5,front_load)-tires.lateralForce(alpha_f-1e-5,front_load))/(2e-5);
    const double fyf0=2.0*model.road_grip_scale*tires.lateralForce(alpha_f,front_load);
    const double fx_over_m=ax-operating.vy*operating.r+fyf0*steer/model.mass;
    const auto body_derivative=[&](const VehicleState& s) {
        const double fyf=2.0*model.road_grip_scale*
            tires.lateralForce(steer-(s.vy+model.a*s.r)/s.vx,front_load);
        const double fyr=2.0*model.road_grip_scale*
            tires.lateralForce((model.b*s.r-s.vy)/s.vx,rear_load);
        return PredictionState{(fyf+fyr)/model.mass-s.vx*s.r,
            s.vy*s.r+fx_over_m-fyf*steer/model.mass,
            s.r,(model.a*fyf-model.b*fyr)/model.iz,
            s.vx*std::sin(s.psi)+s.vy*std::cos(s.psi),
            s.vx*std::cos(s.psi)-s.vy*std::sin(s.psi)};
    };
    const auto linear=discretizeErrorModel(model,tires,operating,steer,ax,small_dt);
    const PredictionState working_state{operating.vy,operating.vx,operating.psi,
        operating.r,operating.y,operating.x};
    const auto predicted=matrixTimesState(linear.a,working_state);
    const auto actual_derivative=body_derivative(operating);
    {
        auto tire_state = operating;
        tire_state.alpha_front = 0.015;
        tire_state.alpha_rear = -0.005;
        constexpr double front_force_bias = 200.0;
        constexpr double rear_force_bias = -60.0;
        const auto dynamic = discretizeErrorModel(model, tires, tire_state, steer, ax, small_dt,
            0.15, 0.40, 0.30, front_force_bias, rear_force_bias);
        auto x = working_state;
        x[6] = tire_state.alpha_front;
        x[7] = tire_state.alpha_rear;
        auto next = matrixTimesState(dynamic.a, x);
        for (int i = 0; i < prediction_state_count; ++i)
            next[i] += dynamic.b[i] * steer + dynamic.c[i];
        const double mu = model.road_grip_scale;
        const double front_force = 2 * mu * tires.lateralForce(x[6] / mu, front_load) +
                                   front_force_bias;
        const double rear_force = 2 * mu * tires.lateralForce(x[7] / mu, rear_load) +
                                  rear_force_bias;
        const double yaw_accel = (model.a * front_force * std::cos(steer) -
                                  model.b * rear_force) / model.iz;
        const double yaw_steer_derivative = -model.a * front_force * std::sin(steer) / model.iz;
        constexpr double jacobian_dt = 1e-7;
        const auto derivative = discretizeErrorModel(model, tires, tire_state, steer, ax, jacobian_dt,
            0.15, 0.40, 0.30, front_force_bias, rear_force_bias);
        require(std::abs(derivative.b[3] / jacobian_dt - yaw_steer_derivative) < 0.01 &&
                std::abs(next[3] - x[3] - small_dt * yaw_accel) < 1e-7 &&
                next[6] > x[6] && next[7] < x[7] && dynamic.b[6] > 0.0,
                "Dynamic tire prediction must start at measured slip and relax toward instantaneous slip");
    }
    const double slip_f=steer-(operating.vy+model.a*operating.r)/operating.vx;
    const double slip_r=(model.b*operating.r-operating.vy)/operating.vx;
    require(std::abs(tires.lateralForce(slip_f,front_load) +
                     tires.lateralForce(-slip_f,front_load)) < 1e-8,
            "Nonlinear tire force must be odd in slip");
    const double fyf_nl=2.0*model.road_grip_scale*tires.lateralForce(slip_f,front_load);
    const double fyr_nl=2.0*model.road_grip_scale*tires.lateralForce(slip_r,rear_load);
    const double expected_vy_dot=(fyf_nl+fyr_nl)/model.mass-operating.vx*operating.r;
    const double expected_yaw_dot=(model.a*fyf_nl-model.b*fyr_nl)/model.iz;
    require(std::abs(predicted[0]+linear.b[0]*steer+linear.c[0]-
                     working_state[0]-small_dt*expected_vy_dot)<1e-6 &&
            std::abs(predicted[3]+linear.b[3]*steer+linear.c[3]-
                     working_state[3]-small_dt*expected_yaw_dot)<1e-6,
            "Nonlinear affine model misses its operating-point tire forces");
    for (int equation=0;equation<6;++equation)
        require(std::abs(predicted[equation]+linear.b[equation]*steer+
                    linear.c[equation]-working_state[equation]-
                    small_dt*actual_derivative[equation])<1e-6,
                "Three-DOF affine model misses its CarSim operating point");
    for (int variable=0;variable<4;++variable) {
        auto plus=operating,minus=operating;
        double* p=variable==0 ? &plus.vy : variable==1 ? &plus.vx :
                  variable==2 ? &plus.psi : &plus.r;
        double* n=variable==0 ? &minus.vy : variable==1 ? &minus.vx :
                  variable==2 ? &minus.psi : &minus.r;
        *p+=epsilon;
        *n-=epsilon;
        const auto fp=body_derivative(plus),fm=body_derivative(minus);
        for (int equation=0;equation<6;++equation) {
            const double numerical=(fp[equation]-fm[equation])/(2.0*epsilon);
            const double jac=(linear.a[equation][variable]-
                (equation==variable ? 1.0 : 0.0))/small_dt;
            require(std::abs(numerical-jac)<0.01,
                "Three-DOF six-state Jacobian disagrees with nonlinear equations");
        }
    }
    const auto linear_steer=discretizeErrorModel(model,tires,operating,
        steer+epsilon,ax,small_dt);
    const double numeric_steer=2.0*model.road_grip_scale*
        (tires.lateralForce(alpha_f+epsilon,front_load)*(steer+epsilon)-
         tires.lateralForce(alpha_f-epsilon,front_load)*(steer-epsilon))/(2.0*epsilon);
    require(std::isfinite(linear.b[1]) && std::isfinite(linear_steer.b[1]) &&
            std::abs(linear.b[1])>1e-10 &&
            std::abs(linear.a[4][2])>1e-8,
            "Three-DOF longitudinal steering projection or global kinematics missing");
    const double control_dt = 0.05;
    const auto tustin = discretizeErrorModel(model, tires, operating,
                                             steer, ax, control_dt);
    // At a finite sampling period, verify the bilinear identity
    // (I-T*A/2) A_d = I+T*A/2 using numerical continuous Jacobians.
    for (int variable = 0; variable < 6; ++variable) {
        VehicleState plus = operating, minus = operating;
        double* positive = variable == 0 ? &plus.vy : variable == 1 ? &plus.vx :
            variable == 2 ? &plus.psi : variable == 3 ? &plus.r :
            variable == 4 ? &plus.y : &plus.x;
        double* negative = variable == 0 ? &minus.vy : variable == 1 ? &minus.vx :
            variable == 2 ? &minus.psi : variable == 3 ? &minus.r :
            variable == 4 ? &minus.y : &minus.x;
        *positive += epsilon;
        *negative -= epsilon;
        const auto fp = body_derivative(plus), fm = body_derivative(minus);
        for (int row = 0; row < 6; ++row) {
            const double jac = (fp[row] - fm[row]) / (2.0 * epsilon);
            double lhs = tustin.a[row][variable];
            for (int inner = 0; inner < 6; ++inner) {
                VehicleState high = operating, low = operating;
                double* hp = inner == 0 ? &high.vy : inner == 1 ? &high.vx :
                    inner == 2 ? &high.psi : inner == 3 ? &high.r :
                    inner == 4 ? &high.y : &high.x;
                double* lp = inner == 0 ? &low.vy : inner == 1 ? &low.vx :
                    inner == 2 ? &low.psi : inner == 3 ? &low.r :
                    inner == 4 ? &low.y : &low.x;
                *hp += epsilon; *lp -= epsilon;
                lhs -= 0.5 * control_dt *
                    (body_derivative(high)[row] - body_derivative(low)[row]) /
                    (2.0 * epsilon) * tustin.a[inner][variable];
            }
            require(std::abs(lhs - (row == variable ? 1.0 : 0.0) -
                             0.5 * control_dt * jac) < 1e-6,
                    "Tustin discretization does not satisfy its matrix identity");
        }
    }
    const double continuous_input[6] = {cf / model.mass,
        -numeric_steer / model.mass, 0.0, model.a * cf / model.iz,
        0.0, 0.0};
    for (int row = 0; row < 6; ++row) {
        double lhs = tustin.b[row];
        for (int inner = 0; inner < 6; ++inner) {
            VehicleState high = operating, low = operating;
            double* hp = inner == 0 ? &high.vy : inner == 1 ? &high.vx :
                inner == 2 ? &high.psi : inner == 3 ? &high.r :
                inner == 4 ? &high.y : &high.x;
            double* lp = inner == 0 ? &low.vy : inner == 1 ? &low.vx :
                inner == 2 ? &low.psi : inner == 3 ? &low.r :
                inner == 4 ? &low.y : &low.x;
            *hp += epsilon; *lp -= epsilon;
            lhs -= 0.5 * control_dt *
                (body_derivative(high)[row] - body_derivative(low)[row]) /
                (2.0 * epsilon) * tustin.b[inner];
        }
        require(std::abs(lhs - control_dt * continuous_input[row]) < 1e-6,
                "Tustin input matrix disagrees with the continuous steering Jacobian");
    }
    for (const auto [np,nc] : {std::pair{15,8}, std::pair{30,1}}) {
        auto p = config.mpc;
        p.np = np; p.nc = nc;
        LateralMpc mpc(model, p, ref, tires);
        VehicleState state;
        state.x = 50.0; state.vx = model.vx;
        const auto result = mpc.control(state, 0.0);
        double previous = 0.0;
        bool all_moves_feasible = true;
        for (double command : mpc.lastControlMoves()) {
            all_moves_feasible &= std::abs(command) <= p.max_steer + 1e-12 &&
                std::abs(command - previous) <= p.dt * p.max_steer_rate + 1e-12;
            previous = command;
        }
        require(std::isfinite(result.steer) &&
                std::abs(result.steer) <= p.max_steer + 1e-12 &&
                std::abs(result.steer) <= p.dt * p.max_steer_rate + 1e-12 &&
                mpc.lastControlMoves().size() == static_cast<std::size_t>(nc) &&
                all_moves_feasible &&
                mpc.predictedSteeringAt(np - 1) == mpc.lastControlMoves().back(),
                "Incremental MPC steering, rate, or horizon constraints failed");
        auto adjusted = p;
        adjusted.np += 1;
        adjusted.q_y *= 1.1;
        LateralMpc retuned(model, adjusted, ref, tires);
        retuned.adoptPreviousSolution(mpc);
        require(retuned.lastControlMoves() == mpc.lastControlMoves(),
                "Adaptive MPC did not retain the previous steering sequence");
    }
    {
        auto p = config.mpc;
        p.np = 35; p.nc = 5;
        p.dt = 0.05; // Keep the fixture's 1.75 s preview independent of default.ini.
        // Use the instantaneous-tire fixture here so friction changes the
        // small-slip stiffness as well as the saturation force.
        p.front_relaxation_m = p.rear_relaxation_m = 0.0;
        p.q_y = 700.0;
        p.r_delta = 800.0;
        p.max_steer = 0.42;
        p.max_steer_rate = 65.0 * 3.14159265358979323846 / 180.0;
        VehicleState approaching;
        approaching.x = 50.0;
        approaching.vx = 20.0;
        LateralMpc constant_grip(model, p, ref, tires);
        LateralMpc preview_grip(model, p, ref, tires);
        double furthest_station = approaching.x;
        preview_grip.setRoadGripProfile([&furthest_station](double x) {
            furthest_station = std::max(furthest_station, x);
            return x < 60.0 ? 0.8 : 0.4;
        });
        const auto old_command = constant_grip.control(approaching, 0.0);
        const auto new_command = preview_grip.control(approaching, 0.0);
        double max_move_difference = 0.0;
        for (std::size_t i = 0; i < constant_grip.lastControlMoves().size(); ++i)
            max_move_difference = std::max(max_move_difference,
                std::abs(constant_grip.lastControlMoves()[i] -
                         preview_grip.lastControlMoves()[i]));
        require(old_command.converged && new_command.converged &&
                furthest_station > 60.0 &&
                max_move_difference > 1e-6,
                "MPC did not apply the future road-grip profile inside its horizon");
    }
    {
        auto p = config.mpc;
        p.np = 13; p.nc = 7;
        LateralMpc mpc(model, p, ref, tires);
        VehicleState approaching;
        approaching.vx = 80.0 / 3.6;
        const auto result = mpc.control(approaching, 0.0);
        require(result.converged &&
                result.iterations <= p.admm_max_iterations,
                "Incremental LTV QP failed with curved-path preview");
        p.stage_weight_lambda = 1.0;
        LateralMpc far_weighted(model, p, ref, tires);
        const auto weighted = far_weighted.control(approaching, 0.0);
        double move_difference = 0.0;
        for (std::size_t j = 0; j < mpc.lastControlMoves().size(); ++j)
            move_difference = std::max(move_difference,
                std::abs(mpc.lastControlMoves()[j] - far_weighted.lastControlMoves()[j]));
        require(weighted.converged &&
                move_difference > 1e-8,
                "Prediction-step exponential weights did not enter the QP");
    }
    {
        auto p = config.mpc;
        LateralMpc mpc(model, p, ref, tires);
        VehicleState risky;
        risky.vx = 95.0 / 3.6;
        risky.vy = risky.vx * 0.13;
        risky.alpha_front = 0.8;
        risky.alpha_rear = 0.8;
        risky.fy_front = 2500.0;
        risky.fy_rear = 2500.0;
        const auto result = mpc.control(risky, 0.0);
        require(result.iterations > 0 && result.prediction_failure.empty() &&
                std::isfinite(result.steer) &&
                std::abs(result.steer) <= p.dt * effectiveSteerRate(p, risky.vx) + 1e-12,
                "High sideslip must still produce a feasible MPC move");
        risky.vx = 0.0;
        const auto held = mpc.control(risky, result.steer);
        require(!held.converged && !held.prediction_failure.empty() && held.steer == result.steer,
                "Invalid operating point must hold the previous applied steering");
    }
    std::cout << "PASS: CarSim-based MPC optimization constraints\n";
}

void testIncrementalClosedLoop() {
    const auto path = std::filesystem::temp_directory_path() /
        "mpc_incremental_straight_fixture.par";
    {
        std::ofstream out(path);
        out << "set_description path_id Straight East\nLTARG_TABLE SPLINE_FLAT\n"
               "0, 0\n100, 0\n200, 0\nENDTABLE\n";
    }
    const auto reference = DoubleLaneChange::fromCarSimParsfile(path);
    std::filesystem::remove(path);
    TireTable tires(std::string(PROJECT_SOURCE_DIR) + "/configs/magic_tire.csv");
    auto settings = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/default.ini").mpc;
    settings.np = 20;
    settings.nc = 7;
    // This synthetic plant uses instantaneous cornering forces.
    settings.front_relaxation_m = settings.rear_relaxation_m = 0.0;
    VehicleParameters vehicle;
    vehicle.mass = 1501.0; vehicle.iz = 2192.089539;
    vehicle.a = 1.049562958; vehicle.b = 1.860437042;
    vehicle.road_grip_scale = 0.8; vehicle.vx = 15.0;
    const auto [front_load, rear_load] = staticWheelLoads(vehicle);
    const double cf = 2.0 * vehicle.road_grip_scale *
        tires.corneringStiffness(front_load);
    const double cr = 2.0 * vehicle.road_grip_scale *
        tires.corneringStiffness(rear_load);
    {
        auto weighted = settings;
        weighted.r_delta_change = 100000.0;
        VehicleState state;
        LateralMpc steady(vehicle, weighted, reference, tires);
        LateralMpc turning(vehicle, weighted, reference, tires);
        const auto a = steady.control(state, 0.0, 0.0);
        const auto b = turning.control(state, 0.0, 0.005);
        require(a.converged && b.converged && b.steer > a.steer + 1e-6,
                "Steering rate change penalty did not use the preceding applied increment");
    }
    for (const double speed : {30.0 / 3.6, 15.0}) {
        vehicle.vx = speed;
        LateralMpc mpc(vehicle, settings, reference, tires);
        VehicleState state;
        state.vx = speed;
        state.y = 0.10;
        double previous_steer = 0.0;
        double previous_increment = 0.0;
        int converged_steps = 0;
        for (int step = 0; step < 40; ++step) {
            const auto command = mpc.control(state, previous_steer, previous_increment);
            require(std::isfinite(command.steer) &&
                    std::abs(command.steer) <= settings.max_steer + 1e-12 &&
                    std::abs(command.steer - previous_steer) <=
                        settings.max_steer_rate * settings.dt + 1e-12,
                    "Incremental closed-loop command violates actuator constraints");
            converged_steps += command.converged ? 1 : 0;
            previous_increment = command.steer - previous_steer;
            previous_steer = command.steer;
            for (int substep = 0; substep < 10; ++substep) {
                const double h = settings.dt / 10.0;
                const double fyf = cf * (command.steer -
                    (state.vy + vehicle.a * state.r) / state.vx);
                const double fyr = cr * ((vehicle.b * state.r - state.vy) / state.vx);
                const double vy_dot = (fyf + fyr) / vehicle.mass - state.vx * state.r;
                const double r_dot = (vehicle.a * fyf - vehicle.b * fyr) / vehicle.iz;
                state.x += h * (state.vx * std::cos(state.psi) -
                                state.vy * std::sin(state.psi));
                state.y += h * (state.vx * std::sin(state.psi) +
                                state.vy * std::cos(state.psi));
                state.psi += h * state.r;
                state.vy += h * vy_dot;
                state.r += h * r_dot;
            }
        }
        require(converged_steps >= 35 && std::abs(state.y) < 0.08,
                "Incremental LTV-MPC did not recover a small straight-road offset");
    }
    std::cout << "PASS: incremental LTV-MPC closes the nonlinear three-DOF loop\n";
}
} // namespace
int main() {
    try {
        testReference();
        testTireZeroLoadColumn();
        testConfig();
        testCarSimBrowserResultSync();
        testCarSimRackSteeringLimit();
        testCarSimMpcModel();
        testRlModule();
        testMpcConstraints();
        testIncrementalClosedLoop();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
