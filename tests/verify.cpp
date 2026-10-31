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
    return ref;
}
void testConfig() {
    auto config = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/default.ini");
    require(config.plant == "carsim" && config.carsim.sync_browser_result,
            "CarSim default configuration incorrect");
    auto schedule = config.mpc;
    schedule.speed_rate_enabled = true;
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
    MpcResult grip_only;
    grip_only.grip_limited = true;
    require(!optimizationFailed(grip_only), "Intentional grip tracking counted as solver failure");
    grip_only.iterations = 2500;
    require(optimizationFailed(grip_only), "Nonconverged QP hidden by grip tracking");
    grip_only.iterations = 0;
    grip_only.prediction_failure = "invalid prediction";
    require(optimizationFailed(grip_only), "Prediction failure hidden by grip tracking");
    overrideConfig(config, "mpc.np", "30");
    overrideConfig(config, "mpc.nc", "10");
    overrideConfig(config, "weights.q_y", "250");
    overrideConfig(config, "rl.train_speed_min_kmh", "36");
    validateConfig(config);
    require(config.mpc.np == 30 && config.mpc.nc == 10 && config.mpc.q_y == 250 &&
            std::abs(config.rl.train_speed_min - 10.0) < 1e-12,
            "MPC or RL override incorrect");
    const auto saved = configToIni(config);
    require(saved.find("[carsim]") != std::string::npos &&
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
    require(inherited.mpc.np == 21 && inherited.mpc.q_y == 432 && inherited.mpc.nonlinear_tire,
            "Inherited values or overlay incorrect");
    config.mpc.np = 22;
    config.mpc.q_y = 321;
    write(fixture / "base.ini", configToIni(config));
    inherited = loadConfig(fixture / "overlay.ini");
    require(inherited.mpc.np == 22 && inherited.mpc.q_y == 321,
            "Changed base configuration was not reloaded");
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
    require(std::filesystem::equivalent(root.parent_path(), temp),
            "Unexpected Browser sync test directory");
    std::filesystem::remove_all(root);
    std::cout << "PASS: CarSim Browser sync archives cache and rejects changed source\n";
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
    require(std::abs(quantizeNormalizedNp(0.10, 25, 18, 40) - 2.0/15.0) < 1e-12 &&
            std::abs(quantizeNormalizedNp(-0.20, 25, 18, 40) + 1.0/7.0) < 1e-12 &&
            quantizeNormalizedNp(0.0, 25, 18, 40) == 0.0,
            "Critic Np quantization does not match the executed integer action");
    require(trackingSafetyAcceptable(1.06, 0, 1.0, 0) &&
            !trackingSafetyAcceptable(1.08, 0, 1.0, 0) &&
            !trackingSafetyAcceptable(1.0, 1, 1.0, 0),
            "Heading and grip protection checkpoint gates incorrect");
    auto config = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/default.ini");
    const auto fast = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/carsim_focus_fast.ini");
    require(fast.rl.decision_steps * fast.mpc.dt <= 0.2 + 1e-12 &&
            std::abs(std::pow(fast.rl.discount_gamma, 5.0) - 0.98) < 1e-9 &&
            fast.rl.action_profile == "compact" && !fast.rl.low_speed_blend,
            "Fast focus policy timing, discount or action profile incorrect");
    const auto guarded_fast = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/carsim_focus_fast_guarded.ini");
    require(guarded_fast.rl.high_speed_blend &&
            policyActionBlend(80.0, guarded_fast.rl) == 1.0 &&
            policyActionBlend(100.0, guarded_fast.rl) == 0.0,
            "Fast policy high-speed guard is not configured correctly");
    const auto fixed60 = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/carsim_fixed60_np_focus.ini");
    require(fixed60.rl.strict_tracking_selection && fixed60.rl.action_profile == "compact" &&
            std::abs(fixed60.rl.train_speed_min * 3.6 - 60.0) < 1e-9 &&
            std::abs(fixed60.rl.train_speed_max * 3.6 - 60.0) < 1e-9 &&
            fixed60.rl.np_exploration_scale > 1.0,
            "Fixed-60 training configuration does not emphasize Np and tracking gates");
    auto invalid_discount = fast;
    invalid_discount.rl.discount_gamma = 1.1;
    bool discount_rejected = false;
    try { validateConfig(invalid_discount); } catch (const std::invalid_argument&) { discount_rejected = true; }
    require(discount_rejected, "Invalid policy discount must be rejected");
    overrideConfig(config, "rl.mode", "train");
    auto relative_reward = config.rl;
    relative_reward.relative_advantage = true;
    require(std::abs(pairedRewardAdvantage(-1.0, -2.0, relative_reward) - 0.5) < 1e-12 &&
            std::abs(pairedRewardAdvantage(-50.0, -100.0, relative_reward) - 0.5) < 1e-12 &&
            pairedRewardAdvantage(1000.0, -2.0, relative_reward) == 8.0 &&
            std::abs(pairedRewardAdvantage(-1.0, -2.0, config.rl) - 0.1) < 1e-12,
            "Paired reward scaling, bounded floor or clipping incorrect");
    auto direct_reward = relative_reward;
    direct_reward.direct_block_reward = true;
    require(std::abs(trainingBlockReward(-10.0, -2.0, direct_reward) + 1.0) < 1e-12 &&
            std::abs(trainingBlockReward(-1.0, -2.0, relative_reward) - 0.5) < 1e-12,
            "Direct block reward or paired reward mode incorrect");
    auto blend_config = config.rl;
    blend_config.low_speed_blend = true;
    require(lowSpeedActionBlend(50.0, blend_config) == 0.0 &&
            lowSpeedActionBlend(60.0, blend_config) == 0.0 &&
            std::abs(lowSpeedActionBlend(70.0, blend_config) - 0.5) < 1e-12 &&
            lowSpeedActionBlend(80.0, blend_config) == 1.0 &&
            lowSpeedActionBlend(120.0, blend_config) == 1.0,
            "Low speed action blend endpoints incorrect");
    blend_config.low_speed_blend = false;
    blend_config.high_speed_blend = true;
    require(policyActionBlend(80.0, blend_config) == 1.0 &&
            policyActionBlend(85.0, blend_config) == 1.0 &&
            std::abs(policyActionBlend(90.0, blend_config) - 0.5) < 1e-12 &&
            policyActionBlend(95.0, blend_config) == 0.0 &&
            policyActionBlend(120.0, blend_config) == 0.0,
            "High speed policy blend endpoints incorrect");
    blend_config.low_speed_blend = true;
    blend_config.high_speed_blend = false;
    overrideConfig(config, "rl.episodes", "200");
    require(config.rl.episodes == 200 && config.rl.mode == "train",
            "RL training overrides did not select 200 episodes");
    overrideConfig(config, "rl.min_np", "10");
    overrideConfig(config, "rl.max_np", "50");
    validateConfig(config);
    auto reference = testReference();
    VehicleState state;
    state.vx = 18.0;
    const auto observation = makeRlObservation(state, reference, config.mpc.np,
                                                config.distance_m, 30.0, config.rl);
    auto augmented_config = config.rl;
    augmented_config.action_profile = "compact";
    augmented_config.augmented_observation = true;
    const auto augmented = makeRlObservation(state, reference, config.mpc.np,
        config.distance_m, 30.0, augmented_config, neutralRlAction(augmented_config),
        0.8, 0.21, true, -0.105, 0.42);
    require(std::abs(augmented[12] - 0.5) < 1e-12 && augmented[13] == 1.0 &&
            std::abs(augmented[14] + 0.25) < 1e-12,
            "Augmented steering, grip mode or planned steering observation incorrect");
    auto agent = createRlPolicy(config.rl);
    auto legacy_compact_config = augmented_config;
    legacy_compact_config.augmented_observation = false;
    auto legacy_compact = createRlPolicy(legacy_compact_config);
    auto legacy_obs = observation;
    legacy_obs[12] = legacy_obs[13] = legacy_obs[14] = 0.0;
    const auto old_actor_file = std::filesystem::temp_directory_path() / "mpc_compact_actor_migration.txt";
    for (int i = 0; i < 100; ++i) {
        const auto sampled = legacy_compact->act(legacy_obs, true);
        legacy_compact->observe(legacy_obs, sampled, -0.2, legacy_obs, false);
    }
    legacy_compact->save(old_actor_file);
    auto augmented_agent = createRlPolicy(augmented_config);
    augmented_agent->warmStart(old_actor_file);
    auto new_obs = legacy_obs; new_obs[12] = 0.7; new_obs[13] = 1.0; new_obs[14] = -0.3;
    const auto old_action = legacy_compact->act(legacy_obs, false);
    const auto new_action = augmented_agent->act(new_obs, false);
    require(old_action.np == new_action.np && std::abs(old_action.q_y - new_action.q_y) < 1e-9 &&
            std::abs(old_action.q_psi - new_action.q_psi) < 1e-9 &&
            std::abs(old_action.r_delta - new_action.r_delta) < 1e-9,
            "Old compact Actor migration changed behavior before new features are trained");
    bool old_direct_load_rejected = false;
    try { augmented_agent->load(old_actor_file); } catch (const std::runtime_error&) { old_direct_load_rejected = true; }
    require(old_direct_load_rejected, "Augmented policy accepted old observation format without warm-start migration");
    auto grip_config = legacy_compact_config;
    grip_config.grip_demand_clip = 4.0;
    auto grip_agent = createRlPolicy(grip_config);
    grip_agent->warmStart(old_actor_file);
    const auto migrated_action = grip_agent->act(legacy_obs, false);
    require(migrated_action.np == old_action.np && std::abs(migrated_action.q_y - old_action.q_y) < 1e-9,
            "Grip observation warm start changed the actor for identical inputs");
    bool grip_old_rejected = false;
    try { grip_agent->load(old_actor_file); } catch (const std::runtime_error&) { grip_old_rejected = true; }
    require(grip_old_rejected, "Grip observation policy accepted old schema without warm start");
    const auto grip_file = std::filesystem::temp_directory_path() / "mpc_grip_actor_migration.txt";
    grip_agent->save(grip_file);
    bool legacy_grip_rejected = false;
    try { legacy_compact->load(grip_file); } catch (const std::runtime_error&) { legacy_grip_rejected = true; }
    require(legacy_grip_rejected, "Old observation policy accepted grip schema");
    std::filesystem::remove(grip_file);
    std::filesystem::remove(old_actor_file);
    blend_config.action_profile = "compact";
    auto blend_agent = createRlPolicy(blend_config);
    auto low_observation = observation;
    low_observation[0] = 50.0 / 108.0;
    const auto blend_neutral = neutralRlAction(blend_config);
    for (int i = 0; i < 20; ++i) {
        const auto sample = blend_agent->act(low_observation, true);
        require(sample.np == blend_neutral.np && std::abs(sample.q_y - blend_neutral.q_y) < 1e-9 &&
                std::abs(sample.q_psi - blend_neutral.q_psi) < 1e-9 &&
                std::abs(sample.r_delta - blend_neutral.r_delta) < 1e-9,
                "Low speed blend did not keep neutral action under exploration");
    }
    auto high_agent = createRlPolicy(guarded_fast.rl);
    auto high_observation = observation;
    high_observation[0] = 100.0 / 108.0;
    const auto high_neutral = neutralRlAction(guarded_fast.rl);
    for (int i = 0; i < 20; ++i) {
        const auto sample = high_agent->act(high_observation, true);
        require(sample.np == high_neutral.np && std::abs(sample.q_y - high_neutral.q_y) < 1e-9 &&
                std::abs(sample.q_psi - high_neutral.q_psi) < 1e-9 &&
                std::abs(sample.r_delta - high_neutral.r_delta) < 1e-9,
                "High speed guard did not keep neutral action under exploration");
    }
    const auto action = agent->act(observation, false);
    const auto neutral = neutralRlAction(config.rl);
    auto np_only_config = config.rl;
    np_only_config.action_profile = "np_only";
    auto np_only_agent = createRlPolicy(np_only_config);
    const auto fixed_weights = neutralRlAction(np_only_config);
    for (int i = 0; i < 100; ++i) {
        const auto sample = np_only_agent->act(observation, true);
        require(std::abs(sample.q_y - fixed_weights.q_y) < 1e-9 &&
                std::abs(sample.q_psi - fixed_weights.q_psi) < 1e-9 &&
                std::abs(sample.q_vy - fixed_weights.q_vy) < 1e-9 &&
                std::abs(sample.q_r - fixed_weights.q_r) < 1e-9 &&
                std::abs(sample.r_steer - fixed_weights.r_steer) < 1e-9 &&
                std::abs(sample.r_delta - fixed_weights.r_delta) < 1e-9,
                "Np-only exploration changed Q/R weights");
    }
    const auto np_policy_file = std::filesystem::temp_directory_path() / "mpc_np_only_policy_test.txt";
    np_only_agent->save(np_policy_file);
    auto reloaded_np_agent = createRlPolicy(np_only_config);
    reloaded_np_agent->load(np_policy_file);
    auto transfer_config = np_only_config;
    transfer_config.action_profile = "compact";
    auto transfer_agent = createRlPolicy(transfer_config);
    transfer_agent->warmStart(np_policy_file);
    const auto compact_start = transfer_agent->act(observation, false);
    require(compact_start.np == fixed_weights.np &&
            std::abs(compact_start.q_y - fixed_weights.q_y) < 1e-9 &&
            std::abs(compact_start.r_delta - fixed_weights.r_delta) < 1e-9,
            "Np-only warm start changed compact policy center");
    bool profile_rejected = false;
    try { agent->load(np_policy_file); } catch (const std::runtime_error&) { profile_rejected = true; }
    require(profile_rejected, "Full-action policy accepted an Np-only policy file");
    std::filesystem::remove(np_policy_file);
    require(action.np == config.rl.nominal_np &&
            std::abs(action.q_y - neutral.q_y) < 1e-9 &&
            std::abs(action.r_delta - neutral.r_delta) < 1e-9,
            "Untrained policy must start at the independent RL action center");
    require(action.np >= config.rl.min_np && action.np <= config.rl.max_np,
            "RL action escaped horizon bounds");
    const auto tuned = applyRlAction(config.mpc, action, config.rl);
    require(tuned.nc == config.mpc.nc && tuned.np == action.np,
            "RL action changed Nc or failed to set Np");
    MpcTuningAction selected = action;
    selected.q_y = config.rl.weight_max[0];
    selected.q_psi = config.rl.weight_min[1];
    selected.q_vy = config.rl.weight_max[2];
    selected.q_r = config.rl.weight_min[3];
    selected.r_steer = config.rl.weight_max[4];
    selected.r_delta = config.rl.weight_min[5];
    const auto weighted = applyRlAction(config.mpc, selected, config.rl);
    require(std::abs(weighted.q_y - selected.q_y) < 1e-9 &&
            std::abs(weighted.q_psi - selected.q_psi) < 1e-9 &&
            std::abs(weighted.q_vy - selected.q_vy) < 1e-9 &&
            std::abs(weighted.q_r - selected.q_r) < 1e-9 &&
            std::abs(weighted.r_steer - selected.r_steer) < 1e-9 &&
            std::abs(weighted.r_delta - selected.r_delta) < 1e-9,
            "RL action did not apply absolute Q/R weights");
    auto edited_fixed = config.mpc;
    edited_fixed.np += 5;
    edited_fixed.q_y *= 3.0;
    edited_fixed.r_delta *= 0.5;
    edited_fixed.terminal_scale *= 2.0;
    const auto independent = applyRlAction(edited_fixed, selected, config.rl);
    require(independent.np == weighted.np &&
            independent.q_y == weighted.q_y &&
            independent.r_delta == weighted.r_delta &&
            independent.terminal_scale == weighted.terminal_scale,
            "Editing fixed Np/Q/R changed DDPG-applied Np/Q/R");
    bool scale_rejected = false;
    try {
        auto invalid = selected;
        invalid.r_delta = config.rl.weight_max[5] + 0.01;
        (void)applyRlAction(config.mpc, invalid, config.rl);
    } catch (const std::invalid_argument&) { scale_rejected = true; }
    require(scale_rejected, "RL adapter accepted an out-of-range R weight");
    const double reward0 = scalarizedRlReward(0.0, 0.0, 0.0,
                                               config.rl.min_np, config.mpc, config.rl);
    const double reward_y = scalarizedRlReward(config.rl.reward.ey_scale_m,
                                                0.0, 0.0, config.rl.min_np,
                                                config.mpc, config.rl);
    std::array<double, 11> reward_terms{};
    const double diagnosed_reward = scalarizedRlReward(config.rl.reward.ey_scale_m,
        0.0, 0.0, config.rl.min_np, config.mpc, config.rl, 0.0, -1, {}, {}, 0.0, 0.0, false, &reward_terms);
    double reward_term_sum = 0.0;
    for (double term : reward_terms) reward_term_sum += term;
    require(std::abs(diagnosed_reward - reward_y) < 1e-12 &&
            std::abs(diagnosed_reward + config.rl.reward.reward_scale * reward_term_sum) < 1e-12,
            "Reward diagnostic terms must reproduce the actual unclipped reward");
    const double reward_psi = scalarizedRlReward(0.0,
        config.rl.reward.epsi_scale_rad, 0.0, config.rl.min_np,
        config.mpc, config.rl);
    require(std::abs(reward0) < 1e-12 &&
            std::abs(reward_y +
                config.rl.reward.reward_scale * config.rl.reward.w_y) < 1e-12 &&
            std::abs(reward_psi +
                config.rl.reward.reward_scale * config.rl.reward.w_psi) < 1e-12,
            "RL scalarized reward does not apply normalized weighted terms");
    const double reward_peak = scalarizedRlReward(
        2.0 * config.rl.reward.ey_scale_m, 0.0, 0.0,
        config.rl.min_np, config.mpc, config.rl);
    const double peak_excess = std::max(0.0,
        2.0 * config.rl.reward.ey_scale_m - config.rl.reward.peak_threshold_m) /
        config.rl.reward.peak_threshold_m;
    const double smooth_error_two = (std::sqrt(5.0) - 1.0) /
                                    (std::sqrt(2.0) - 1.0);
    require(std::abs(reward_peak + config.rl.reward.reward_scale *
                    (smooth_error_two * config.rl.reward.w_y +
                     config.rl.reward.w_peak * peak_excess * peak_excess)) < 1e-12,
            "RL lateral reward must penalize tracking and peak error");
    const double reward_large = scalarizedRlReward(3.0, 0.0, 0.0,
        config.rl.min_np, config.mpc, config.rl);
    const double reward_larger = scalarizedRlReward(4.0, 0.0, 0.0,
        config.rl.min_np, config.mpc, config.rl);
    require(reward_larger < reward_large,
            "RL reward must distinguish severe lateral tracking errors");
    const double one_rate_step = config.mpc.max_steer_rate * config.mpc.dt;
    const double reward_rate = scalarizedRlReward(0.0, 0.0,
        one_rate_step, config.rl.min_np, config.mpc, config.rl, 0.0);
    const double reward_no_rate = scalarizedRlReward(0.0, 0.0,
        one_rate_step, config.rl.min_np, config.mpc, config.rl,
        one_rate_step);
    require(std::abs((reward_no_rate - reward_rate) -
        config.rl.reward.reward_scale * config.rl.reward.w_steer_rate) < 1e-12,
        "RL steering-rate penalty must use the previous command");
    const double reward_change = scalarizedRlReward(0.0, 0.0, 0.0,
        config.rl.min_np, config.mpc, config.rl, 0.0, -1, selected);
    const double reward_grip = scalarizedRlReward(0.0, 0.0, 0.0,
        config.rl.min_np, config.mpc, config.rl, 0.0, -1,
        MpcTuningAction{}, MpcTuningAction{}, 0.0, 0.0, true);
    require(reward_change < reward0 && reward_grip < reward0,
            "RL reward must penalize pointless weight movement and grip limits");
    state.x = 65.0;
    state.y = reference.at(state.x).y + 2.0;
    const auto error_two = makeRlObservation(state, reference, config.mpc.np,
        config.distance_m, 30.0, config.rl);
    state.y = reference.at(state.x).y + 3.0;
    const auto error_three = makeRlObservation(state, reference, config.mpc.np,
        config.distance_m, 30.0, config.rl);
    require(error_three[1] > error_two[1],
            "RL observation must distinguish large lateral errors");
    state.y = reference.at(state.x).y;
    const auto near_bend = makeRlObservation(state, reference, config.mpc.np,
        config.distance_m, 30.0, config.rl);
    require(std::abs(near_bend[6] - observation[6]) > 1e-4 ||
            std::abs(near_bend[7] - observation[7]) > 1e-4,
            "RL observation ignores preview curvature and its variation");
    const auto low_mu = makeRlObservation(state, reference, config.mpc.np,
        config.distance_m, 30.0, config.rl, MpcTuningAction{}, 0.3);
    const auto high_mu = makeRlObservation(state, reference, config.mpc.np,
        config.distance_m, 30.0, config.rl, MpcTuningAction{}, 0.9);
    require(low_mu[16] < high_mu[16] && low_mu[9] > high_mu[9],
            "RL observation must distinguish grip and speed-curvature demand");
    auto high_demand_state = state;
    high_demand_state.vx = 100.0;
    const auto old_demand = makeRlObservation(high_demand_state, reference, config.mpc.np,
        config.distance_m, 30.0, legacy_compact_config, MpcTuningAction{}, 0.3);
    const auto new_demand = makeRlObservation(high_demand_state, reference, config.mpc.np,
        config.distance_m, 30.0, grip_config, MpcTuningAction{}, 0.3);
    require(old_demand[9] == 3.0 && new_demand[9] == 4.0,
            "Grip demand schema must apply its configured observation limit");
    auto fixed_mu_config = config;
    fixed_mu_config.rl.train_mu_min = 0.8;
    fixed_mu_config.rl.train_mu_max = 0.8;
    validateConfig(fixed_mu_config);
    const auto fixed_mu_observation = makeRlObservation(state, reference,
        config.mpc.np, config.distance_m, 30.0, fixed_mu_config.rl,
        MpcTuningAction{}, 0.8);
    require(std::isfinite(fixed_mu_observation[16]) &&
            fixed_mu_observation[16] == 0.0,
            "Fixed-mu training must avoid zero-width normalization");
    const auto fixed_mu_file = std::filesystem::temp_directory_path() /
                               "mpc_fixed_mu_policy_test.txt";
    auto fixed_mu_agent = createRlPolicy(fixed_mu_config.rl);
    const auto fixed_neutral = neutralRlAction(fixed_mu_config.rl);
    for (int i = 0; i < 70; ++i)
        fixed_mu_agent->observe(fixed_mu_observation, fixed_neutral,
                                0.1, fixed_mu_observation, false);
    const auto fixed_action = fixed_mu_agent->act(fixed_mu_observation, false);
    fixed_mu_agent->save(fixed_mu_file);
    createRlPolicy(fixed_mu_config.rl)->load(fixed_mu_file);
    bool fixed_mu_rejected = false;
    try { createRlPolicy(config.rl)->load(fixed_mu_file); }
    catch (const std::runtime_error&) { fixed_mu_rejected = true; }
    require(fixed_mu_rejected,
            "A fixed-mu policy must reject a different mu training range");
    auto widened = createRlPolicy(config.rl);
    widened->warmStart(fixed_mu_file);
    auto wider_observation = fixed_mu_observation;
    wider_observation[16] = 2.0 * (0.8 - config.rl.train_mu_min) /
        (config.rl.train_mu_max - config.rl.train_mu_min) - 1.0;
    const auto widened_action = widened->act(wider_observation, false);
    require(widened_action.np == fixed_action.np &&
            std::abs(widened_action.q_y - fixed_action.q_y) < 1e-10 &&
            std::abs(widened_action.q_psi - fixed_action.q_psi) < 1e-10 &&
            std::abs(widened_action.r_delta - fixed_action.r_delta) < 1e-10,
            "Fixed-mu warm start changed the actor at its original grip");
    const auto state_file = std::filesystem::temp_directory_path() /
                            "mpc_rl_training_state_test.txt";
    fixed_mu_agent->saveTrainingState(state_file);
    auto resumed = createRlPolicy(fixed_mu_config.rl);
    resumed->loadTrainingState(state_file);
    const auto resumed_action = resumed->act(fixed_mu_observation, false);
    require(resumed_action.np == fixed_action.np &&
            std::abs(resumed_action.q_y - fixed_action.q_y) < 1e-10,
            "Training state did not restore the actor");
    fixed_mu_agent->observe(fixed_mu_observation, fixed_neutral, 0.2,
                            fixed_mu_observation, false);
    resumed->observe(fixed_mu_observation, fixed_neutral, 0.2,
                     fixed_mu_observation, false);
    const auto after_original = fixed_mu_agent->act(fixed_mu_observation, false);
    const auto after_resume = resumed->act(fixed_mu_observation, false);
    require(after_original.np == after_resume.np &&
            std::abs(after_original.q_y - after_resume.q_y) < 1e-10 &&
            std::abs(after_original.q_psi - after_resume.q_psi) < 1e-10,
            "Restored optimizer, replay or RNG produced a different next update");
    auto mismatched_state = fixed_mu_config.rl;
    mismatched_state.reward.w_y += 0.1;
    bool mismatch_rejected = false;
    try { createRlPolicy(mismatched_state)->loadTrainingState(state_file); }
    catch (const std::runtime_error&) { mismatch_rejected = true; }
    require(mismatch_rejected, "Training state accepted incompatible reward settings");
    std::filesystem::remove(state_file);
    std::filesystem::remove(fixed_mu_file);
    state.vx = 12.0;
    const auto slow_bend = makeRlObservation(state, reference, config.mpc.np,
        config.distance_m, 30.0, config.rl);
    require(std::abs(slow_bend[0] - near_bend[0]) > 1e-3 &&
            std::abs(slow_bend[9] - near_bend[9]) > 1e-4,
            "RL observation ignores speed-dependent lateral demand");
    const auto policy_file = std::filesystem::temp_directory_path() /
                             "mpc_ddpg_policy_test.txt";
    agent->setTrainingEpisode(1, 200);
    require(std::abs(agent->explorationStdDev() - 0.25) < 1e-12,
            "DDPG exploration must start with the configured training scale");
    agent->setTrainingEpisode(200, 200);
    require(std::abs(agent->explorationStdDev() - 0.04) < 1e-12,
            "DDPG exploration must decay without disappearing");
    agent->save(policy_file);
    auto restored = createRlPolicy(config.rl);
    restored->load(policy_file);
    const auto reloaded = restored->act(observation, false);
    require(reloaded.np == action.np &&
            std::abs(reloaded.q_y - action.q_y) < 1e-12 &&
            std::abs(reloaded.q_psi - action.q_psi) < 1e-12 &&
            std::abs(reloaded.q_vy - action.q_vy) < 1e-12 &&
            std::abs(reloaded.q_r - action.q_r) < 1e-12 &&
            std::abs(reloaded.r_steer - action.r_steer) < 1e-12 &&
            std::abs(reloaded.r_delta - action.r_delta) < 1e-12,
            "Saved policy gives different deterministic Np/Q/R actions");
    config.mpc.np = 25;
    config.mpc.q_y = 240.0;
    config.mpc.r_delta = 100.0;
    validateConfig(config);
    auto after_fixed_edit = createRlPolicy(config.rl);
    after_fixed_edit->load(policy_file);
    const auto same_action = after_fixed_edit->act(observation, false);
    const auto same_tuning = applyRlAction(config.mpc, same_action, config.rl);
    require(same_action.np == action.np &&
            same_tuning.q_y == tuned.q_y &&
            same_tuning.r_delta == tuned.r_delta,
            "Same policy changed Np/Q/R after editing fixed MPC parameters");
    auto changed_rl = config.rl;
    changed_rl.terminal_scale += 1.0;
    bool changed_rl_rejected = false;
    try { createRlPolicy(changed_rl)->load(policy_file); }
    catch (const std::runtime_error&) { changed_rl_rejected = true; }
    require(changed_rl_rejected, "Policy accepted a changed RL terminal cost setting");
    auto td3_config = config.rl;
    td3_config.algorithm = "td3";
    auto td3_policy = createRlPolicy(td3_config);
    bool cross_algorithm_rejected = false;
    try { td3_policy->load(policy_file); }
    catch (const std::runtime_error&) { cross_algorithm_rejected = true; }
    require(cross_algorithm_rejected,
            "TD3 accepted a DDPG policy with different training semantics");
    auto compact_config = config.rl;
    compact_config.action_profile = "compact";
    auto compact_policy = createRlPolicy(compact_config);
    const auto compact_neutral = neutralRlAction(compact_config);
    for (int i = 0; i < 70; ++i)
        compact_policy->observe(observation, compact_neutral,
                                0.1, observation, false);
    const auto compact_action = compact_policy->act(observation, false);
    require(std::abs(compact_action.q_vy - compact_neutral.q_vy) < 1e-10 &&
            std::abs(compact_action.q_r - compact_neutral.q_r) < 1e-10 &&
            std::abs(compact_action.r_steer - compact_neutral.r_steer) < 1e-10,
            "Compact policy changed an inactive MPC weight");
    const auto compact_file = std::filesystem::temp_directory_path() /
                              "mpc_compact_policy_test.txt";
    compact_policy->save(compact_file);
    createRlPolicy(compact_config)->load(compact_file);
    bool compact_as_full_rejected = false;
    try { createRlPolicy(config.rl)->load(compact_file); }
    catch (const std::runtime_error&) { compact_as_full_rejected = true; }
    require(compact_as_full_rejected,
            "Full-action DDPG accepted a compact-action policy");
    std::filesystem::remove(compact_file);
    const auto td3_file = std::filesystem::temp_directory_path() /
                          "mpc_td3_policy_test.txt";
    td3_policy->save(td3_file);
    auto td3_restored = createRlPolicy(td3_config);
    td3_restored->load(td3_file);
    require(td3_restored->act(observation, false).np == config.rl.nominal_np,
            "TD3 policy round trip changed the neutral action");
    std::filesystem::remove(td3_file);
    std::filesystem::remove(policy_file);
    {
        std::ofstream legacy(policy_file);
        legacy << "DDPG_MPC_V14 15 10 40 0.6 1.666 0.7 1.428 0.3 0.9\n";
    }
    bool legacy_rejected = false;
    try { after_fixed_edit->load(policy_file); }
    catch (const std::runtime_error&) { legacy_rejected = true; }
    std::filesystem::remove(policy_file);
    require(legacy_rejected, "Legacy multiplier policy was treated as an absolute-action policy");
    {
        std::ofstream old_absolute(policy_file);
        old_absolute << "DDPG_MPC_V15_ABSOLUTE 15 10 40\n";
    }
    bool old_controller_rejected = false;
    try { after_fixed_edit->load(policy_file); }
    catch (const std::runtime_error&) { old_controller_rejected = true; }
    std::filesystem::remove(policy_file);
    require(old_controller_rejected,
            "Old absolute-action policy was accepted by incremental LTV-MPC");
    bool rejected = false;
    config.rl.min_np = config.mpc.nc - 1;
    try { validateConfig(config); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "RL min_np below Nc must be rejected");
    config.rl.min_np = config.mpc.nc;
    config.rl.reward.ey_scale_m = 0.0;
    rejected = false;
    try { validateConfig(config); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Zero RL reward scale must be rejected");
    std::cout << "PASS: RL bounds, MPC parameter adapter, policy round trip\n";
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
    const double cf=2.0*model.road_grip_scale*tires.corneringStiffness(front_load);
    const double cr=2.0*model.road_grip_scale*tires.corneringStiffness(rear_load);
    const double fyf0=cf*(steer-(operating.vy+model.a*operating.r)/operating.vx);
    const double fx_over_m=ax-operating.vy*operating.r+fyf0*steer/model.mass;
    const auto body_derivative=[&](const VehicleState& s) {
        const double fyf=cf*(steer-(s.vy+model.a*s.r)/s.vx);
        const double fyr=cr*((model.b*s.r-s.vy)/s.vx);
        return State6{(fyf+fyr)/model.mass-s.vx*s.r,
            s.vy*s.r+fx_over_m-fyf*steer/model.mass,
            s.r,(model.a*fyf-model.b*fyr)/model.iz,
            s.vx*std::sin(s.psi)+s.vy*std::cos(s.psi),
            s.vx*std::cos(s.psi)-s.vy*std::sin(s.psi)};
    };
    const auto linear=discretizeErrorModel(model,tires,operating,steer,ax,small_dt);
    const State6 working_state{operating.vy,operating.vx,operating.psi,
        operating.r,operating.y,operating.x};
    const auto predicted=matrixTimesState(linear.a,working_state);
    const auto actual_derivative=body_derivative(operating);
    const double slip_f=steer-(operating.vy+model.a*operating.r)/operating.vx;
    const double slip_r=(model.b*operating.r-operating.vy)/operating.vx;
    require(std::abs(tires.lateralForce(slip_f,front_load) +
                     tires.lateralForce(-slip_f,front_load)) < 1e-8,
            "Nonlinear tire force must be odd in slip");
    const double fyf_nl=2.0*model.road_grip_scale*tires.lateralForce(slip_f,front_load);
    const double fyr_nl=2.0*model.road_grip_scale*tires.lateralForce(slip_r,rear_load);
    const auto nonlinear=discretizeErrorModel(model,tires,operating,steer,ax,small_dt,true,0.15);
    const auto nonlinear_prediction=matrixTimesState(nonlinear.a,working_state);
    const double expected_vy_dot=(fyf_nl+fyr_nl)/model.mass-operating.vx*operating.r;
    const double expected_yaw_dot=(model.a*fyf_nl-model.b*fyr_nl)/model.iz;
    require(std::abs(nonlinear_prediction[0]+nonlinear.b[0]*steer+nonlinear.c[0]-
                     working_state[0]-small_dt*expected_vy_dot)<1e-6 &&
            std::abs(nonlinear_prediction[3]+nonlinear.b[3]*steer+nonlinear.c[3]-
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
    const double numeric_steer=(cf*(steer+epsilon-(operating.vy+
        model.a*operating.r)/operating.vx)*(steer+epsilon)-
        cf*(steer-epsilon-(operating.vy+model.a*operating.r)/
        operating.vx)*(steer-epsilon))/(2.0*epsilon);
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
        p.np = 13; p.nc = 7;
        LateralMpc mpc(model, p, ref, tires);
        VehicleState approaching;
        approaching.vx = 80.0 / 3.6;
        const auto result = mpc.control(approaching, 0.0);
        require(result.converged && !result.grip_limited &&
                result.iterations <= p.admm_max_iterations,
                "Incremental LTV QP failed with curved-path preview");
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
    VehicleParameters vehicle;
    vehicle.mass = 1501.0; vehicle.iz = 2192.089539;
    vehicle.a = 1.049562958; vehicle.b = 1.860437042;
    vehicle.road_grip_scale = 0.8; vehicle.vx = 15.0;
    const auto [front_load, rear_load] = staticWheelLoads(vehicle);
    const double cf = 2.0 * vehicle.road_grip_scale *
        tires.corneringStiffness(front_load);
    const double cr = 2.0 * vehicle.road_grip_scale *
        tires.corneringStiffness(rear_load);
    for (const double speed : {30.0 / 3.6, 15.0}) {
        vehicle.vx = speed;
        LateralMpc mpc(vehicle, settings, reference, tires);
        VehicleState state;
        state.vx = speed;
        state.y = 0.10;
        double previous_steer = 0.0;
        int converged_steps = 0;
        for (int step = 0; step < 40; ++step) {
            const auto command = mpc.control(state, previous_steer);
            require(std::isfinite(command.steer) &&
                    std::abs(command.steer) <= settings.max_steer + 1e-12 &&
                    std::abs(command.steer - previous_steer) <=
                        settings.max_steer_rate * settings.dt + 1e-12,
                    "Incremental closed-loop command violates actuator constraints");
            converged_steps += command.converged ? 1 : 0;
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
void testThreeControllerPlots() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output = std::filesystem::temp_directory_path() /
        ("mpc_three_plots_" + std::to_string(unique));
    std::vector<SimulationRecord> fixed(2), ddpg(2), td3(2);
    for (int i = 0; i < 2; ++i) {
        fixed[i].time = ddpg[i].time = td3[i].time = static_cast<double>(i);
        fixed[i].s.x = ddpg[i].s.x = td3[i].s.x = static_cast<double>(i);
        fixed[i].s.y = 0.1 * i;
        ddpg[i].s.y = 0.2 * i;
        td3[i].s.y = 0.3 * i;
        fixed[i].reference.y = 0.4 * i;
        fixed[i].prediction_horizon = 13;
        ddpg[i].prediction_horizon = 15;
        td3[i].prediction_horizon = 17;
    }
    exportThreeComparisonVisuals(output, fixed, ddpg, td3);
    std::ifstream in(output / "01_lateral_tracking.svg", std::ios::binary);
    const std::string svg{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    in.close();
    require(svg.find("Reference") != std::string::npos &&
            svg.find("Fixed MPC") != std::string::npos &&
            svg.find("DDPG") != std::string::npos &&
            svg.find("TD3") != std::string::npos,
            "Three-controller plot omitted a trajectory or its legend");
    require(std::filesystem::exists(output / "08_prediction_horizon.svg") &&
            std::filesystem::exists(output / "10_r_delta_value.svg"),
            "Three-controller report omitted an MPC or weight plot");
    for (const auto& entry : std::filesystem::directory_iterator(output))
        std::filesystem::remove(entry.path());
    std::filesystem::remove(output);
    std::cout << "PASS: three-controller plots contain all controllers\n";
}
} // namespace
int main() {
    try {
        testReference();
        testConfig();
        testCarSimBrowserResultSync();
        testCarSimMpcModel();
        testRlModule();
        testMpcConstraints();
        testIncrementalClosedLoop();
        testThreeControllerPlots();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
