#include "config.h"
#include "carsim_plant.h"
#include "mpc.h"
#include "reference.h"
#include "vehicle.h"
#include "rl.h"
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
    std::filesystem::remove_all(root);
    std::cout << "PASS: CarSim solver echo supplies the MPC vehicle and tire model\n";
}
void testRlModule() {
    auto config = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/default.ini");
    overrideConfig(config, "rl.mode", "train");
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
    auto agent = createRlPolicy(config.rl);
    const auto action = agent->act(observation, false);
    const auto neutral = neutralRlAction(config.rl);
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
    require(std::abs(reward_peak + config.rl.reward.reward_scale *
                    (4.0 * config.rl.reward.w_y +
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
    for (int equation=0;equation<6;++equation)
        require(std::abs(predicted[equation]+linear.b[equation]*steer+
                    linear.c[equation]-working_state[equation]-
                    small_dt*actual_derivative[equation])<1e-9,
                "Paper six-state affine model misses its CarSim operating point");
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
                "Paper six-state Jacobian disagrees with nonlinear equations");
        }
    }
    const auto linear_steer=discretizeErrorModel(model,tires,operating,
        steer+epsilon,ax,small_dt);
    const double numeric_steer=(cf*(steer+epsilon-(operating.vy+
        model.a*operating.r)/operating.vx)*(steer+epsilon)-
        cf*(steer-epsilon-(operating.vy+model.a*operating.r)/
        operating.vx)*(steer-epsilon))/(2.0*epsilon);
    require(std::abs(linear.b[1]/small_dt+numeric_steer/model.mass)<1e-9 &&
            std::isfinite(linear_steer.b[1]) &&
            std::abs(linear.b[1])>1e-10 &&
            std::abs(linear.a[4][2])>1e-8,
            "Paper longitudinal steering projection or global kinematics missing");
    const auto kinematic = discretizeKinematicModel(model, operating, steer, ax, 0.05);
    auto kin_next = matrixTimesState(kinematic.a, working_state);
    for (int d = 0; d < 6; ++d)
        kin_next[d] += kinematic.b[d] * steer + kinematic.c[d];
    require(std::abs(kin_next[0] - operating.vy) < 1e-9 &&
            std::abs(kin_next[1] - (operating.vx + 0.05 * ax)) < 1e-9 &&
            std::abs(kin_next[2] - (operating.psi + 0.05 * operating.r)) < 1e-9 &&
            std::abs(kin_next[3] - operating.r) < 1e-9 &&
            kinematic.b[0] == 0.0 && kinematic.b[3] > 0.0,
            "Measured-sideslip kinematic model misses its CarSim operating point");
    require(dynamicPredictionWeight(30.0 / 3.6) == 0.0 &&
            std::abs(dynamicPredictionWeight(36.0 / 3.6) - 0.5) < 1e-12 &&
            dynamicPredictionWeight(40.0 / 3.6) == 1.0,
            "The model blend must center at 36 km/h");
    for (const auto [np,nc] : {std::pair{15,8}, std::pair{30,1}}) {
        auto p = config.mpc;
        p.np = np; p.nc = nc;
        LateralMpc mpc(model, p, ref, tires);
        VehicleState state;
        state.x = 50.0; state.vx = model.vx;
        const auto result = mpc.control(state, 0.0);
        require(std::isfinite(result.steer) &&
                std::abs(result.steer) <= p.max_steer + 1e-12 &&
                std::abs(result.steer) <= p.dt * p.max_steer_rate + 1e-12 &&
                mpc.lastControlMoves().size() == static_cast<std::size_t>(nc),
                "MPC steering or horizon constraints failed");
    }
    std::cout << "PASS: CarSim-based MPC optimization constraints\n";
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
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
