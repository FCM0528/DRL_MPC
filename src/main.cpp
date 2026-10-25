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
#include <iterator>
#include <memory>
#include <map>
#include <set>
#include <tuple>
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
    name << readableToken(config.experiment) << "_carsim";
    name << "_Np" << (config.rl.mode == "train" ? config.rl.nominal_np : config.mpc.np)
         << "_Nc" << config.mpc.nc;
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
        << "  --rl-mode MODE             fixed | train | eval | compare | eval_grid\n"
        << "  --rl-model PATH            Policy file for eval; filename for train\n"
        << "  --rl-episodes N            Number of DDPG training episodes (default: 200)\n"
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
        else if (key == "--rl-model") overrideConfig(c, "rl.model_file", value);
        else if (key == "--rl-episodes") overrideConfig(c, "rl.episodes", value);
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
                              int np = 0, const MpcTuningAction& action = {}) {
    SimulationRecord row;
    row.time = time;
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
    row.dynamic_model_weight = dynamicPredictionWeight(s.vx);
    row.q_y = action.q_y;
    row.q_psi = action.q_psi;
    row.q_vy = action.q_vy;
    row.q_r = action.q_r;
    row.r_steer = action.r_steer;
    row.r_delta = action.r_delta;
    return row;
}

MpcResult neutralFallback(double previous_steer, const MpcParameters& mpc) {
    // Do not invent an optimized command after a QP failure: return steering
    // toward zero while respecting both the angle and per-sample rate bounds.
    MpcResult result;
    const double max_step = mpc.dt * mpc.max_steer_rate;
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
};

SimulationRun runConfiguredSimulation(const AppConfig& opts) {
        VehicleParameters model_vehicle = opts.vehicle;
        CarSimConfig run_config = opts.carsim;
        run_config.simfile = stageCarSimSimfile(opts.carsim.simfile,
                                                 opts.output / "carsim_native");
        DoubleLaneChange ref = DoubleLaneChange::fromCarSimParsfile(
            opts.output / "carsim_native" / "Run_all.par");
        auto plant = std::make_unique<CarSimPlant>(std::move(run_config));
        const auto carsim_model = readCarSimMpcModel(opts.output / "carsim_native");
        model_vehicle = carsim_model.vehicle;
        model_vehicle.vx = plant->initialState().vx;
        TireTable tires(carsim_model.tire_csv);
        RunOutcome outcome;
        outcome.plant_name = "carsim";
        outcome.controller_mode = opts.rl.mode;
        if (opts.rl.mode == "eval") outcome.policy_file = opts.rl.model_file;
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
        std::map<std::tuple<int, double, double, double, double, double, double>,
                 std::unique_ptr<LateralMpc>> controllers;
        LateralMpc* controller = nullptr;
        std::unique_ptr<RlPolicy> rl_policy;
        int current_np = opts.rl.mode == "eval" ? opts.rl.nominal_np : opts.mpc.np;
        MpcTuningAction current_action = neutralRlAction(opts.rl);
        if (opts.rl.mode != "eval") {
            current_action.np = opts.mpc.np;
            current_action.q_y = opts.mpc.q_y;
            current_action.q_psi = opts.mpc.q_psi;
            current_action.q_vy = opts.mpc.q_vy;
            current_action.q_r = opts.mpc.q_r;
            current_action.r_steer = opts.mpc.r_steer;
            current_action.r_delta = opts.mpc.r_delta;
        }
        if (opts.rl.mode == "eval") {
            rl_policy = createRlPolicy(opts.rl);
            rl_policy->load(opts.rl.model_file);
        }
        try {
            const auto initial_settings = rl_policy
                ? applyRlAction(opts.mpc, current_action, opts.rl) : opts.mpc;
            auto initial = std::make_unique<LateralMpc>(model_vehicle, initial_settings, ref, tires);
            controller = initial.get();
            controllers.emplace(std::make_tuple(current_np, current_action.q_y,
                current_action.q_psi, current_action.q_vy, current_action.q_r,
                current_action.r_steer, current_action.r_delta), std::move(initial));
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
        if (rl_policy) {
            std::cout << "  DDPG Np range=" << opts.rl.min_np << ".." << opts.rl.max_np
                      << ", nominal=" << opts.rl.nominal_np
                      << "; Q/R are absolute policy actions\n";
        } else {
            std::cout << "  Fixed Np=" << opts.mpc.np
                      << " (Tp=" << opts.mpc.np * opts.mpc.dt << " s, Tc="
                      << opts.mpc.nc * opts.mpc.dt << " s)\n"
                      << "  Q=diag(" << opts.mpc.q_y << ", " << opts.mpc.q_psi
                      << ", " << opts.mpc.q_vy << ", " << opts.mpc.q_r << ")"
                      << ", R=" << opts.mpc.r_steer << ", R_delta=" << opts.mpc.r_delta
                      << '\n';
        }
        std::cout << std::flush;

        for (int step = 0; state.x < opts.distance_m && step < max_steps; ++step) {
            if (rl_policy && controller && step % opts.rl.decision_steps == 0) {
                const auto observation = makeRlObservation(
                    state, ref, current_np, opts.distance_m, 30.0, opts.rl,
                    current_action, model_vehicle.road_grip_scale);
                auto action = rl_policy->act(observation, false);
                const auto tuned = applyRlAction(opts.mpc, action, opts.rl);
                {
                    current_action = action;
                    current_np = tuned.np;
                    try {
                        auto& selected = controllers[std::make_tuple(
                            current_np, action.q_y, action.q_psi,
                            action.q_vy, action.q_r,
                            action.r_steer, action.r_delta)];
                        if (!selected)
                            selected = std::make_unique<LateralMpc>(model_vehicle, tuned, ref, tires);
                        controller = selected.get();
                    } catch (const std::exception& e) {
                        if (outcome.first_fallback_reason.empty())
                            outcome.first_fallback_reason = std::string("RL/MPC initialization: ") + e.what();
                        controller = nullptr;
                    }
                }
            }
            const auto start = std::chrono::steady_clock::now();
            MpcResult control;
            bool fallback = !controller;
            if (controller) {
                controller->setGripLimitedMode(grip_mode_active);
                try {
                    control = controller->control(state, prev_delta);
                    if (!std::isfinite(control.steer))
                        throw std::runtime_error("MPC returned nonfinite steering");
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
            if (fallback) control = neutralFallback(prev_delta, opts.mpc);
            const auto stop = std::chrono::steady_clock::now();
            const double solve_ms = std::chrono::duration<double, std::milli>(stop - start).count();
            rows.push_back(recordSample(state, ref, t, control, solve_ms,
                                        true, fallback, current_np, current_action));
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
                                        last_command, 0.0, false, false, current_np, current_action));
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
        return {opts.output, std::move(rows), metrics, outcome};
}

std::string htmlEscape(std::string value) {
    std::string escaped;
    for (char ch : value) {
        if (ch == '&') escaped += "&amp;";
        else if (ch == '<') escaped += "&lt;";
        else if (ch == '>') escaped += "&gt;";
        else if (ch == '"') escaped += "&quot;";
        else escaped += ch;
    }
    return escaped;
}

bool filesEqual(const std::filesystem::path& left, const std::filesystem::path& right) {
    if (std::filesystem::file_size(left) != std::filesystem::file_size(right)) return false;
    std::ifstream a(left, std::ios::binary), b(right, std::ios::binary);
    return std::equal(std::istreambuf_iterator<char>(a), std::istreambuf_iterator<char>(),
                      std::istreambuf_iterator<char>(b));
}

void runCarSimComparison(const AppConfig& opts) {
    const auto policy_source = std::filesystem::absolute(opts.rl.model_file);
    if (!std::filesystem::is_regular_file(policy_source))
        throw std::runtime_error("DDPG policy file does not exist: " + policy_source.string());
    const auto policy_snapshot = opts.output / "ddpg_policy_used.txt";
    std::filesystem::copy_file(policy_source, policy_snapshot,
                               std::filesystem::copy_options::overwrite_existing);
    auto policy_preflight = createRlPolicy(opts.rl);
    policy_preflight->load(policy_snapshot);
    std::string training_plant = "unknown";
    const auto native_training_config = policy_source.parent_path() / "effective_config.ini";
    const auto converted_training_config = policy_source.parent_path() / "v14_training_config.ini";
    const auto source_training_config = std::filesystem::is_regular_file(native_training_config)
        ? native_training_config : converted_training_config;
    if (std::filesystem::is_regular_file(source_training_config)) {
        std::filesystem::copy_file(source_training_config,
                                   opts.output / "policy_training_config.ini",
                                   std::filesystem::copy_options::overwrite_existing);
        std::ifstream in(source_training_config);
        std::string line;
        while (std::getline(in, line)) {
            const auto pos = line.find('=');
            if (pos != std::string::npos &&
                line.substr(0, pos).find("plant") != std::string::npos) {
                training_plant = line.substr(pos + 1);
                training_plant.erase(0, training_plant.find_first_not_of(" \t\r"));
                training_plant.erase(training_plant.find_last_not_of(" \t\r") + 1);
                break;
            }
        }
    }
    const auto source_conversion_manifest = policy_source.parent_path() / "conversion_manifest.txt";
    if (std::filesystem::is_regular_file(source_conversion_manifest))
        std::filesystem::copy_file(source_conversion_manifest,
                                   opts.output / "policy_conversion_manifest.txt",
                                   std::filesystem::copy_options::overwrite_existing);
    AppConfig fixed = opts;
    fixed.output = opts.output / "fixed";
    fixed.experiment += "_fixed";
    fixed.rl.mode = "fixed";
    fixed.carsim.sync_browser_result = false;
    const auto fixed_run = runConfiguredSimulation(fixed);

    AppConfig adaptive = opts;
    adaptive.output = opts.output / "adaptive";
    adaptive.experiment += "_ddpg";
    adaptive.rl.mode = "eval";
    adaptive.rl.model_file = policy_snapshot;
    const auto adaptive_run = runConfiguredSimulation(adaptive);
    if (!filesEqual(fixed.output / "carsim_native" / "Run_visualizer.par",
                    adaptive.output / "carsim_native" / "Run_visualizer.par"))
        throw std::runtime_error("CarSim source changed between fixed and DDPG runs; comparison rejected");
    exportComparisonVisuals(opts.output / "comparison", fixed_run.rows, adaptive_run.rows);
    const auto horizons = [](const SimulationRun& run) {
        std::set<int> values;
        for (const auto& row : run.rows)
            if (row.is_command) values.insert(row.prediction_horizon);
        if (values.empty()) throw std::runtime_error("No MPC command samples in comparison");
        return std::array<int, 3>{*values.begin(), *values.rbegin(),
                                  static_cast<int>(values.size())};
    };
    const auto fixed_np = horizons(fixed_run), adaptive_np = horizons(adaptive_run);
    {
        std::ofstream out(opts.output / "comparison_metrics.csv", std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write CarSim comparison metrics");
        out << "controller,completed,rmse_y_m,max_abs_y_m,rmse_heading_deg,"
               "max_steer_command_deg,qp_nonconverged_steps,grip_limited_steps,"
               "fallback_steps,np_min,np_max,np_unique,distance_reached_m\n";
        const auto write = [&](const char* name, const SimulationRun& run,
                               const std::array<int, 3>& np) {
            out << name << ',' << (run.outcome.completed ? 1 : 0) << ','
                << run.metrics.rmse_y << ',' << run.metrics.max_abs_y << ','
                << run.metrics.rmse_heading_deg << ','
                << run.metrics.max_abs_steering_deg << ','
                << run.metrics.qp_nonconverged_steps << ','
                << run.metrics.grip_limited_steps << ','
                << run.metrics.fallback_control_steps << ','
                << np[0] << ',' << np[1] << ',' << np[2] << ','
                << run.rows.back().s.x << '\n';
        };
        write("fixed", fixed_run, fixed_np);
        write("ddpg_adaptive", adaptive_run, adaptive_np);
    }
    {
        std::ofstream out(opts.output / "comparison_report.html", std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write CarSim comparison report");
        out << "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
               "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
               "<title>CarSim 固定 MPC 与 DDPG 自适应 Np/Q/R 对比</title>"
               "<style>body{font:16px 'Microsoft YaHei',sans-serif;color:#183148;background:#edf3f8;"
               "max-width:1200px;margin:30px auto;padding:0 20px}a{color:#176eb8}"
               "table{border-collapse:collapse;background:white;width:100%;margin:16px 0}"
               "td,th{border:1px solid #cbd9e5;padding:9px;text-align:right}"
               "td:first-child,th:first-child{text-align:left}.plot{background:white;padding:12px;"
               "margin:18px 0;border-radius:8px}.plot img{width:100%}"
               "p{line-height:1.7}</style><h1>CarSim 固定 MPC 与 DDPG 自适应 Np/Q/R</h1>"
               "<p>两次闭环使用同一份 CarSim 工况快照，结果分别归档。"
               "<a href=\"fixed/report.html\">固定 Np 报告</a> · "
               "<a href=\"adaptive/report.html\">DDPG 报告</a> · "
               "<a href=\"comparison_metrics.csv\">对比数据 CSV</a>。"
               "CarSim Browser 最近结果对应 DDPG 运行；固定时域的原生回放保存在固定目录。</p>"
               "<p>DDPG 策略来源：" << htmlEscape(policy_source.string())
            << "；归档副本：<a href=\"ddpg_policy_used.txt\">ddpg_policy_used.txt</a>。"
               "来源配置记录的训练 plant：" << htmlEscape(training_plant) << "。"
               "本次仅加载既有策略在当前 CarSim 工况评估，没有重新训练；"
               "MPC 在 32–40 km/h 平滑切换运动学与动力学预测模型；"
               "DDPG 在全部实测车速范围调节 Np/Q/R；"
               "训练车辆与工况是否相同须由策略来源配置核对。";
        if (std::filesystem::exists(opts.output / "policy_training_config.ini"))
            out << "<a href=\"policy_training_config.ini\">查看训练配置</a>。";
        if (std::filesystem::exists(opts.output / "policy_conversion_manifest.txt"))
            out << "<a href=\"policy_conversion_manifest.txt\">查看旧策略绝对动作迁移记录</a>。";
        out << "</p><table><thead><tr><th>控制器</th><th>完成</th><th>横向 RMSE (m)</th>"
               "<th>最大横向误差 (m)</th><th>航向 RMSE (°)</th><th>Np 范围</th>"
               "<th>Np 不同取值</th><th>受限控制周期</th><th>后备控制周期</th></tr></thead><tbody>";
        const auto row = [&](const char* name, const SimulationRun& run,
                             const std::array<int, 3>& np) {
            out << "<tr><td>" << name << "</td><td>"
                << (run.outcome.completed ? "是" : "否") << "</td><td>"
                << std::fixed << std::setprecision(4) << run.metrics.rmse_y
                << "</td><td>" << run.metrics.max_abs_y << "</td><td>"
                << run.metrics.rmse_heading_deg << "</td><td>" << np[0] << "–"
                << np[1] << "</td><td>" << np[2] << "</td><td>"
                << run.metrics.grip_limited_steps << "</td><td>"
                << run.metrics.fallback_control_steps << "</td></tr>";
        };
        row("固定 Np", fixed_run, fixed_np);
        row("DDPG 自适应 Np/Q/R", adaptive_run, adaptive_np);
        out << "</tbody></table>";
        for (const char* file : {"01_lateral_tracking.svg", "02_heading_tracking.svg",
                 "03_lateral_error.svg", "04_heading_error.svg", "05_steering.svg",
                 "06_yaw_rate.svg", "07_speed.svg", "08_prediction_horizon.svg",
                 "11_model_weight.svg"})
            out << "<div class=\"plot\"><img loading=\"lazy\" src=\"comparison/"
                << file << "\" alt=\"" << file << "\"></div>";
        out << "<h2>DDPG 实际应用的 Q/R 数值</h2><p>每个权重的数值还可在 "
               "<a href=\"adaptive/simulation.csv\">自适应运行 CSV</a> 中核查。</p>";
        for (const char* file : {"09_q_y_value.svg", "09_q_psi_value.svg",
                                 "09_q_vy_value.svg", "09_q_r_value.svg",
                                 "10_r_steer_value.svg", "10_r_delta_value.svg"})
            out << "<div class=\"plot\"><img loading=\"lazy\" src=\"adaptive/"
                << file << "\" alt=\"" << file << "\"></div>";
        out << "</html>";
    }
    std::cout << "CarSim comparison: "
              << std::filesystem::absolute(opts.output / "comparison_report.html").string()
              << '\n';
}

int main(int argc, char** argv) {
    try {
        const AppConfig opts = parseArgs(argc, argv);
        if (opts.rl.mode == "train") {
            runRlTraining(opts);
            showReport(opts.output / "rl_report.html", opts.open_report);
            return 0;
        }
        if (opts.rl.mode == "eval_grid") {
            runRlHeldoutEvaluation(opts);
            return 0;
        }
        if (opts.rl.mode == "compare") {
            runCarSimComparison(opts);
            showReport(opts.output / "comparison_report.html", opts.open_report);
            return 0;
        }
        const auto run = runConfiguredSimulation(opts);
        showReport(run.output / "report.html", opts.open_report);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
