#include "config.h"
#include "mpc.h"
#include "reference.h"
#include "vehicle.h"
#include "visualization.h"
#include "rl.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

namespace {
std::string readableToken(const std::string& value) {
    std::string token;
    for (unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_') {
            token += static_cast<char>(ch);
        } else if (!token.empty() && token.back() != '_') {
            token += '_';
        }
    }
    while (!token.empty() && token.back() == '_') token.pop_back();
    return token.empty() ? "unnamed" : token;
}

std::filesystem::path namedOutputFolder(const AppConfig& config) {
    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream name;
    const auto& mode = config.rl.mode;
    name << "mpc_fixed";
    if (config.experiment != "ddpg_adaptive" && config.experiment != "path_tracking")
        name << '_' << readableToken(config.experiment);
    name << "_carsim";
    if (mode == "fixed") name << "_Np" << config.mpc.np;
    name << "_Nc" << config.mpc.nc;
    name << '_' << std::put_time(&local, "%Y%m%d-%H%M%S");
    const auto base = config.output / name.str();
    std::filesystem::create_directories(config.output);
    for (int suffix = 1;; ++suffix) {
        const auto candidate = suffix == 1 ? base
            : std::filesystem::path(base.string() + "_" + std::to_string(suffix));
        if (std::filesystem::create_directory(candidate)) return candidate;
        if (!std::filesystem::exists(candidate))
            throw std::runtime_error("Cannot reserve unique output directory: " + candidate.string());
    }
}

void help() {
    std::cout
        << "CarSim / MPC / Double Lane Change (C++17)\n"
        << "Usage: MPC_DoubleLaneChange [options]\n\n"
        << "  --config PATH              INI configuration (default: configs/default.ini)\n"
        << "  --plant carsim             CarSim vehicle model\n"
        << "  --np N                     Prediction horizon [steps]\n"
        << "  --nc N                     Control horizon [steps], 1 <= Nc <= Np\n"
        << "  --dt SECONDS               Sampling period\n"
        << "  --q-y V                    Q diagonal: lateral error\n"
        << "  --q-psi V                  Q diagonal: heading error\n"
        << "  --q-vy V                   Q diagonal: lateral velocity\n"
        << "  --q-r V                    Q diagonal: yaw rate error\n"
        << "  --r-steer V                R: feedforward steering deviation\n"
        << "  --r-delta V                Steering increment penalty\n"
        << "  --set SECTION.KEY=VALUE    Override ANY INI parameter (repeatable)\n"
        << "  --out FOLDER               Output folder (creates a unique run subfolder)\n"
        << "  --name TEXT                Experiment name\n"
        << "  --distance METERS          End of simulation X range\n"
        << "  --steer-limit-deg V        Front-wheel angle bound in degrees\n"
        << "  --rate-limit-deg-s V       Front-wheel rate bound in degrees/s\n"
        << "  --no-open                  Do not open HTML automatically\n"
        << "  --rl-mode fixed            Run fixed MPC\n"
        << "  --open                     Open HTML automatically (Windows)\n"
        << "  --help                     Show help\n\n"
        << "Examples:\n"
        << "  MPC_DoubleLaneChange --config configs/default.ini\n"
        << "  MPC_DoubleLaneChange --np 40 --nc 10 --out results/nc10\n"
        << "  MPC_DoubleLaneChange --set weights.q_y=200 --set weights.r_delta=700\n";
}

AppConfig parseArgs(int argc, char** argv) {
    std::filesystem::path config_path;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") { help(); std::exit(0); }
        if (key == "--config") {
            if (i + 1 == argc) throw std::invalid_argument("Missing value after --config");
            config_path = argv[++i];
        }
    }
    if (config_path.empty()) config_path = locateDefaultConfig(argv[0]);
    AppConfig c = loadConfig(config_path);
    // After loading INI, apply all CLI options in their actual order.
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--config") { ++i; continue; }
        if (key == "--no-open") { c.open_report = false; continue; }
        if (key == "--open") { c.open_report = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("Missing value after " + key);
        const std::string value = argv[++i];
        if (key == "--set") {
            const auto pos = value.find('=');
            if (pos == std::string::npos)
                throw std::invalid_argument("--set expects SECTION.KEY=VALUE");
            auto setting = value.substr(0, pos);
            std::transform(setting.begin(), setting.end(), setting.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            overrideConfig(c, value.substr(0, pos), value.substr(pos + 1));
        } else if (key == "--plant") overrideConfig(c, "simulation.plant", value);
        else if (key == "--np") overrideConfig(c, "mpc.np", value);
        else if (key == "--nc") overrideConfig(c, "mpc.nc", value);
        else if (key == "--dt") overrideConfig(c, "mpc.dt_s", value);
        else if (key == "--q-y") overrideConfig(c, "weights.q_y", value);
        else if (key == "--q-psi") overrideConfig(c, "weights.q_psi", value);
        else if (key == "--q-vy") overrideConfig(c, "weights.q_vy", value);
        else if (key == "--q-r") overrideConfig(c, "weights.q_r", value);
        else if (key == "--r-steer") overrideConfig(c, "weights.r_steer", value);
        else if (key == "--r-delta") overrideConfig(c, "weights.r_delta", value);
        else if (key == "--out") {
            overrideConfig(c, "simulation.output", value);
        }
        else if (key == "--name") overrideConfig(c, "experiment.name", value);
        else if (key == "--distance") overrideConfig(c, "simulation.distance_m", value);
        else if (key == "--rl-mode") overrideConfig(c, "rl.mode", value);
        else if (key == "--steer-limit-deg") overrideConfig(c, "constraints.max_steer_deg", value);
        else if (key == "--rate-limit-deg-s") overrideConfig(c, "constraints.max_steer_rate_deg_s", value);
        else throw std::invalid_argument("Unknown CLI option: " + key + " (try --help)");
    }
    validateConfig(c);
    c.output = namedOutputFolder(c);
    return c;
}

SimulationRecord recordSample(const VehicleState& s,
                              const DoubleLaneChange& reference,
                              double time,
                              const MpcResult& command, double solve_ms,
                              bool command_sample, bool fallback_control = false,
                              int np = 0, const MpcTuningAction& action = {},
                              double road_grip = 0.0) {
    SimulationRecord row;
    row.time = time;
    row.road_grip = road_grip;
    row.s = s;
    row.reference = reference.at(s.x);
    row.lateral_error = s.y - row.reference.y;
    row.heading_error = angleDifference(s.psi, row.reference.psi);
    row.steer = command.steer;
    row.yaw_rate_ref = s.vx * row.reference.curvature;
    row.qp_iterations = command.iterations;
    row.qp_primal_residual = command.primal_residual;
    row.qp_dual_residual = command.dual_residual;
    row.qp_converged = command.converged;
    row.grip_limited_control = command.grip_limited;
    row.qp_solve_ms = solve_ms;
    row.is_command = command_sample;
    row.fallback_control = fallback_control;
    row.prediction_horizon = np;
    row.q_y = action.q_y;
    row.q_psi = action.q_psi;
    row.q_vy = action.q_vy;
    row.q_r = action.q_r;
    row.r_steer = action.r_steer;
    row.r_delta = action.r_delta;
    return row;
}

MpcResult neutralFallback(double previous_steer, const MpcParameters& mpc, double vx) {
    // Do not invent an optimized command after a QP failure: return steering
    // toward zero while respecting both the angle and per-sample rate bounds.
    MpcResult result;
    const double max_step = mpc.dt * effectiveSteerRate(mpc, vx);
    const double lower = std::max(-mpc.max_steer, previous_steer - max_step);
    const double upper = std::min(mpc.max_steer, previous_steer + max_step);
    result.steer = std::clamp(0.0, lower, upper);
    return result;
}

bool recordableState(const VehicleState& s) {
    // Stop before overflow would corrupt CSV statistics and SVG coordinates.
    constexpr double bound = 1.0e6;
    return std::isfinite(s.x) && std::isfinite(s.y) && std::isfinite(s.psi) &&
           std::isfinite(s.vx) && std::isfinite(s.ax) &&
           std::isfinite(s.vy) && std::isfinite(s.r) &&
           std::abs(s.x) < bound && std::abs(s.y) < bound &&
           std::abs(s.psi) < bound && std::abs(s.vx) < bound &&
           std::abs(s.ax) < bound &&
           std::abs(s.vy) < bound && std::abs(s.r) < bound;
}

void showReport(const std::filesystem::path& report, bool open_report) {
    const auto absolute = std::filesystem::absolute(report);
    std::cout << "Report: " << absolute.string() << '\n';
#ifdef _WIN32
    if (open_report) {
        const HINSTANCE opened = ShellExecuteW(nullptr, L"open", absolute.c_str(),
                                                nullptr, nullptr, SW_SHOWNORMAL);
        if (reinterpret_cast<std::intptr_t>(opened) <= 32)
            std::cerr << "Cannot auto-open browser. Open Report path manually.\n";
    }
#else
    (void)open_report;
#endif
}
}  // namespace

struct SimulationRun {
    std::filesystem::path output;
    std::vector<SimulationRecord> rows;
    SimulationMetrics metrics;
    RunOutcome outcome;
    double road_grip_scale = 0.0;
    bool variable_road_grip = false;
};

SimulationRun runConfiguredSimulation(const AppConfig& opts) {
        VehicleParameters model_vehicle = opts.vehicle;
        CarSimConfig run_config = opts.carsim;
        run_config.simfile = stageCarSimSimfile(opts.carsim.simfile,
                                                 opts.output / "carsim_native");
        extendCarSimStationStop(opts.output / "carsim_native" / "Run_all.par", opts.distance_m);
        DoubleLaneChange ref = DoubleLaneChange::fromCarSimParsfile(
            opts.output / "carsim_native" / "Run_all.par");
        auto plant = std::make_unique<CarSimPlant>(std::move(run_config));
        const auto carsim_model = readCarSimMpcModel(opts.output / "carsim_native");
        model_vehicle = carsim_model.vehicle;
        model_vehicle.road_grip_scale = carsim_model.road_grip.at(plant->initialState().x);
        model_vehicle.vx = plant->initialState().vx;
        TireTable tires(carsim_model.tire_csv);
        RunOutcome outcome;
        outcome.plant_name = "carsim";
        outcome.variable_road_grip = carsim_model.road_grip.variable();
        outcome.reference_source = "CarSim LTARG_TABLE (straight east road)";
        outcome.carsim_info = readCarSimRunInfo(opts.output / "carsim_native" / "Run_all.par");
        if (!outcome.carsim_info.speed_target_constant.empty() &&
            !carsim_model.speed_target_kmh.empty() &&
            std::stod(outcome.carsim_info.speed_target_constant) !=
                std::stod(carsim_model.speed_target_kmh))
            throw std::runtime_error("CarSim speed target differs between Parsfile and solver echo");
        outcome.carsim_info.speed_target_constant = carsim_model.speed_target_kmh;
        outcome.model_front_stiffness_n_per_rad = carsim_model.front_stiffness_n_per_rad;
        outcome.model_rear_stiffness_n_per_rad = carsim_model.rear_stiffness_n_per_rad;
        outcome.requested_distance_m = opts.distance_m;
        std::unique_ptr<LateralMpc> fixed_controller;
        LateralMpc* controller = nullptr;
        const int current_np = opts.mpc.np;
        MpcTuningAction current_action;
        current_action.np = opts.mpc.np;
        current_action.q_y = opts.mpc.q_y;
        current_action.q_psi = opts.mpc.q_psi;
        current_action.q_vy = opts.mpc.q_vy;
        current_action.q_r = opts.mpc.q_r;
        current_action.r_steer = opts.mpc.r_steer;
        current_action.r_delta = opts.mpc.r_delta;
        try {
            fixed_controller = std::make_unique<LateralMpc>(model_vehicle, opts.mpc, ref, tires);
            controller = fixed_controller.get();
        } catch (const std::exception& e) {
            outcome.first_fallback_reason = std::string("MPC initialization: ") + e.what();
        }
        VehicleState state = plant->initialState();
        double t = 0.0;
        double prev_delta = 0.0;
        bool grip_mode_active = false;
        std::vector<SimulationRecord> rows;
        const double step_budget_speed = state.vx;
        const int max_steps = static_cast<int>(2.5 * opts.distance_m /
                               (std::max(1.0, step_budget_speed) * opts.mpc.dt)) + 1000;
        rows.reserve(static_cast<std::size_t>(max_steps));
        std::cout << "Running: " << opts.experiment << '\n'
                  << "  CarSim initial vx="
                  << 3.6 * step_budget_speed << " km/h, Ts=" << opts.mpc.dt
                  << " s, Nc=" << opts.mpc.nc << '\n';
        std::cout << "  Fixed Np=" << opts.mpc.np
                  << " (Tp=" << opts.mpc.np * opts.mpc.dt << " s, Tc="
                  << opts.mpc.nc * opts.mpc.dt << " s)\n"
                  << "  Q=diag(" << opts.mpc.q_y << ", " << opts.mpc.q_psi
                  << ", " << opts.mpc.q_vy << ", " << opts.mpc.q_r << ")"
                  << ", R=" << opts.mpc.r_steer << ", R_delta=" << opts.mpc.r_delta
                  << '\n';
        std::cout << std::flush;

        for (int step = 0; state.x < opts.distance_m && step < max_steps; ++step) {
            model_vehicle.road_grip_scale = carsim_model.road_grip.at(state.x);
            if (controller) controller->setRoadGrip(model_vehicle.road_grip_scale);
            const auto start = std::chrono::steady_clock::now();
            MpcResult control;
            bool fallback = !controller;
            if (controller) {
                controller->setGripLimitedMode(grip_mode_active);
                try {
                    control = controller->control(state, prev_delta);
                    if (!std::isfinite(control.steer))
                        throw std::runtime_error("MPC returned nonfinite steering");
                    if (!control.prediction_failure.empty() && outcome.first_fallback_reason.empty())
                        outcome.first_fallback_reason = std::string("MPC prediction: ") +
                            control.prediction_failure + " at X=" + std::to_string(state.x) + " m";
                    if (control.iterations > 0 && !control.converged && outcome.first_fallback_reason.empty())
                        outcome.first_fallback_reason = "MPC QP did not converge at X=" +
                            std::to_string(state.x) + " m";
                } catch (const std::exception& e) {
                    if (outcome.first_fallback_reason.empty())
                        outcome.first_fallback_reason = std::string(e.what()) +
                            " at X=" + std::to_string(state.x) + " m";
                    // A solver exception must not turn the steering toward zero
                    // while the original path still requires a lateral response.
                    try {
                        control = controller->enterGripLimitedMode(state, prev_delta);
                    } catch (const std::exception&) {
                        controller = nullptr;
                        fallback = true;
                    }
                }
                if (controller) grip_mode_active = controller->gripLimitedModeActive();
            }
            if (fallback) control = neutralFallback(prev_delta, opts.mpc, state.vx);
            const auto stop = std::chrono::steady_clock::now();
            const double solve_ms = std::chrono::duration<double, std::milli>(stop - start).count();
            rows.push_back(recordSample(state, ref, t, control, solve_ms,
                                        true, fallback, current_np, current_action, model_vehicle.road_grip_scale));
            VehicleState next = state;
            try {
                plant->step(next, control.steer, opts.mpc.dt);
            } catch (const std::exception& e) {
                outcome.stop_reason = std::string("Vehicle integration failed: ") + e.what();
                break;
            }
            if (!recordableState(next)) {
                outcome.stop_reason = "Vehicle state became nonfinite or exceeded numerical safety range";
                break;
            }
            const auto next_reference = ref.at(next.x);
            rows.back().reward = scalarizedRlReward(
                next.y - next_reference.y, angleDifference(next.psi, next_reference.psi),
                control.steer, current_np, opts.mpc, opts.rl, prev_delta, current_np,
                current_action, current_action, next.vy,
                next.r - next.vx * next_reference.curvature, control.grip_limited);
            state = next;
            prev_delta = control.steer;
            t += opts.mpc.dt;
        }
        if (state.x < opts.distance_m) {
            outcome.completed = false;
            if (outcome.stop_reason.empty())
                outcome.stop_reason = "Simulation step limit reached before requested distance";
        } else {
            MpcResult last_command;
            last_command.steer = prev_delta;
            rows.push_back(recordSample(state, ref, t,
                                        last_command, 0.0, false, false, current_np, current_action,
                                        carsim_model.road_grip.at(state.x)));
        }

        // Finalize VS/VSB output before copying it into the Browser cache.
        plant.reset();
        if (opts.carsim.sync_browser_result) {
            try {
                outcome.browser_result_file = syncCarSimBrowserResult(
                    opts.carsim.simfile, opts.output / "carsim_native");
                outcome.browser_result_status = "synchronized";
            } catch (const std::exception& e) {
                outcome.browser_result_status = std::string("failed: ") + e.what();
                std::cerr << "WARNING: CarSim Browser Video/Plot cache was not updated: "
                          << e.what() << '\n';
            }
        } else {
            outcome.browser_result_status = "disabled by carsim.sync_browser_result";
        }

        const auto metrics = exportResults(opts.output, rows, model_vehicle,
                                           opts.mpc, opts.experiment, outcome);
        const auto tire_snapshot = opts.output / "tire_data.csv";
        if (!std::filesystem::exists(tire_snapshot) ||
            !std::filesystem::equivalent(carsim_model.tire_csv, tire_snapshot))
            std::filesystem::copy_file(carsim_model.tire_csv, tire_snapshot,
                                       std::filesystem::copy_options::overwrite_existing);
        {
            std::ofstream out(opts.output / "effective_config.ini", std::ios::binary);
            if (!out) throw std::runtime_error("Cannot write effective_config.ini");
            AppConfig snapshot = opts;
            snapshot.vehicle = model_vehicle;
            snapshot.tire_data_file = "tire_data.csv";
            out << configToIni(snapshot);
        }
        std::cout << std::fixed << std::setprecision(6)
                  << (outcome.completed ? "Finished: X=" : "Partial result: X=")
                  << rows.back().s.x << " m, samples=" << rows.size() << '\n'
                  << "  Lateral RMSE       = " << metrics.rmse_y << " m\n"
                  << "  Max |lateral err|  = " << metrics.max_abs_y << " m\n"
                  << "  Heading RMSE       = " << metrics.rmse_heading_deg << " deg\n"
                  << "  Peak steering      = " << metrics.max_abs_steering_deg << " deg\n"
                  << "  Body vx range       = " << 3.6 * metrics.min_speed_mps << " / "
                  << 3.6 * metrics.max_speed_mps << " km/h\n"
                  << "  Ground speed range  = " << 3.6 * metrics.min_ground_speed_mps << " / "
                  << 3.6 * metrics.max_ground_speed_mps << " km/h\n"
                  << "  Mean / peak QP ms  = " << metrics.mean_qp_time_ms << " / "
                  << metrics.max_qp_time_ms << " ms\n"
                  << "  QP not converged   = " << metrics.qp_nonconverged_steps << " steps\n"
                  << "  Grip-limited track = " << metrics.grip_limited_steps << " steps\n"
                  << "  Fallback steering  = " << metrics.fallback_control_steps << " steps\n"
                  << "  Browser Video/Plot   = " << outcome.browser_result_status << '\n'
                  << "Results: " << std::filesystem::absolute(opts.output).string()
                  << '\n' << std::flush;
        if (!outcome.completed)
            std::cerr << "WARNING: Simulation incomplete: " << outcome.stop_reason
                      << ". Exported actual samples only.\n";
        if (!outcome.first_fallback_reason.empty())
            std::cerr << "WARNING: MPC fallback started: "
                      << outcome.first_fallback_reason << '\n';
        if (metrics.qp_nonconverged_steps > 0)
            std::cerr << "WARNING: Some MPC steps lacked a converged QP result "
                         "(including fallback steps). Check summary.txt.\n";
        const double initial_grip = rows.front().road_grip;
        return {opts.output, std::move(rows), metrics, outcome, initial_grip, carsim_model.road_grip.variable()};
}

int main(int argc, char** argv) {
    try {
        const AppConfig opts = parseArgs(argc, argv);
        const auto run = runConfiguredSimulation(opts);
        showReport(run.output / "report.html", opts.open_report);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
