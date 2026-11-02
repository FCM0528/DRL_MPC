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
    auto config = loadConfig(std::string(PROJECT_SOURCE_DIR) + "/configs/carsim_python_rl.ini");
    const auto neutral = neutralRlAction(config.rl);
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
    const double reward = scalarizedRlReward(0.1, 0.02, 0.01, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral);
    require(std::isfinite(reward) && reward < 0.0, "C++ reward is invalid");
    const double accurate = scalarizedRlReward(0.0, 0.0, 0.0, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral);
    require(std::abs(accurate) < 1e-12, "Zero-loss reward must be zero");
    const double moderate = scalarizedRlReward(0.05, 0.0, 0.0, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral);
    const double large = scalarizedRlReward(0.25, 0.0, 0.0, neutral.np,
        config.mpc, config.rl, 0.0, neutral.np, neutral, neutral);
    require(accurate > moderate && moderate > large &&
            accurate - moderate > 0.2 * config.rl.reward.reward_scale,
            "Reward does not distinguish small tracking errors");
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
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
