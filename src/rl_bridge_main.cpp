// Standalone CarSim/Gymnasium bridge. Existing C++ MPC and RL code is unchanged.
// Protocol (one line per command): RESET | STEP a_np a_qy a_qpsi a_rdelta | CLOSE
// Reply: BRIDGE_RESET <17 observation values>
//        BRIDGE_STEP <17 observation values> <reward> <terminated> <truncated>
//            <x> <ey> <epsi> <steer> <qp_failed> <grip_mode> <n_control_steps> <failed> <executed_np>
// Each process serves one episode. Python spawns another process for reset.
#include "config.h"
#include "carsim_plant.h"
#include "mpc.h"
#include "reference.h"
#include "rl.h"
#include "vehicle.h"
#include "visualization.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

double clip(double v, double lo, double hi) { return std::clamp(v, lo, hi); }

void ensureStopTime(const fs::path& parsfile, double min_stop_s) {
    std::ifstream in(parsfile);
    if (!in) throw std::runtime_error("Cannot read staged Run_all.par");
    std::ostringstream result;
    std::string line;
    int matches = 0;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string keyword;
        fields >> keyword;
        if (keyword == "TSTOP") {
            double existing = 0;
            if (!(fields >> existing) || !std::isfinite(existing))
                throw std::runtime_error("Invalid numeric TSTOP in staged Parsfile");
            result << "TSTOP " << std::setprecision(17)
                   << std::max(existing, min_stop_s) << '\n';
            ++matches;
        } else result << line << '\n';
    }
    if (matches != 1) throw std::runtime_error("Expected exactly one TSTOP in Parsfile");
    std::ofstream out(parsfile, std::ios::trunc);
    out << result.str();
    if (!out) throw std::runtime_error("Could not extend staged TSTOP");
}

double mappedWeight(double x, double lower, double upper) {
    if (!std::isfinite(x) || x < -1.0 || x > 1.0)
        throw std::invalid_argument("Actions must be finite numbers in [-1,1]");
    return std::exp(0.5 * ((1.0 - x) * std::log(lower)
                         + (1.0 + x) * std::log(upper)));
}

// Exactly match src/rl.cpp::DdpgPolicy::decodeNp: piecewise linear action,
// C++ lround (round half away from zero), then bounded integer horizon.
int mappedNp(double a, const RlConfig& rl, int nc) {
    if (!std::isfinite(a) || a < -1.0 || a > 1.0)
        throw std::invalid_argument("Np action must be finite and in [-1,1]");
    if (rl.min_np < nc || rl.min_np >= rl.nominal_np || rl.nominal_np >= rl.max_np)
        throw std::invalid_argument("Require Nc <= min_np < nominal_np < max_np");
    const double extent = a >= 0.0 ? rl.max_np - rl.nominal_np
                                   : rl.nominal_np - rl.min_np;
    const long rounded = std::lround(rl.nominal_np + a * extent);
    return std::clamp(static_cast<int>(rounded), std::max(rl.min_np, nc), rl.max_np);
}

class Episode {
    AppConfig cfg_;
    fs::path dir_;
    std::unique_ptr<DoubleLaneChange> reference_;
    std::unique_ptr<CarSimPlant> plant_;
    CarSimMpcModel model_;
    std::unique_ptr<TireTable> tires_;
    VehicleParameters vehicle_;
    VehicleState state_;
    std::unique_ptr<LateralMpc> controller_; // same MPC warm start across control cycles
    MpcTuningAction current_{};
    double previous_steer_ = 0.0;
    bool grip_ = false;
    bool ended_ = false;
    int total_steps_ = 0;
    int max_steps_ = 0;
    std::ofstream trace_;
    std::vector<SimulationRecord> records_;

    double gripAtCurrentX() const { return model_.road_grip.at(state_.x); }
    double plannedSteer() const {
        if (controller_ && controller_->hasLastSolution() &&
            !controller_->lastControlMoves().empty()) {
            const auto& controls = controller_->lastControlMoves();
            return controls[std::min<std::size_t>(1, controls.size() - 1)];
        }
        return previous_steer_;
    }
public:
    Episode(const AppConfig& input, const fs::path& dir) : cfg_(input), dir_(dir) {
        if (cfg_.rl.min_np < cfg_.mpc.nc || cfg_.rl.min_np >= cfg_.rl.nominal_np || cfg_.rl.nominal_np >= cfg_.rl.max_np)
            throw std::invalid_argument("Require Nc <= min_np < nominal_np < max_np");
        // Only read/modify PRIVATE copies; never modify original CarSim Run Control.
        fs::create_directories(dir_);
        const auto native = dir_ / "carsim_native";
        const auto staged = stageCarSimSimfile(cfg_.carsim.simfile, native);
        auto pars = native / "Run_all.par";
        extendCarSimStationStop(pars, cfg_.distance_m);
        auto source = readCarSimRunInfo(pars);
        const double target_speed = source.speed_target_constant.empty() ? 1.0 :
            std::stod(source.speed_target_constant) / 3.6;
        if (!std::isfinite(target_speed) || target_speed < 0.0)
            throw std::runtime_error("CarSim target speed must be finite and nonnegative");
        const double budget_speed = std::max(1.0, target_speed);
        ensureStopTime(pars, cfg_.distance_m / (budget_speed * 0.75) + 5.0);
        reference_ = std::make_unique<DoubleLaneChange>(
            DoubleLaneChange::fromCarSimParsfile(pars));
        auto run_cfg = cfg_.carsim;
        run_cfg.simfile = staged;
        run_cfg.sync_browser_result = false;
        plant_ = std::make_unique<CarSimPlant>(std::move(run_cfg));
        model_ = readCarSimMpcModel(native);
        state_ = plant_->initialState();
        vehicle_ = model_.vehicle;
        vehicle_.vx = state_.vx;
        tires_ = std::make_unique<TireTable>(model_.tire_csv);
        cfg_.mpc.np = cfg_.rl.nominal_np;
        current_ = neutralRlAction(cfg_.rl);
        current_.np = cfg_.rl.nominal_np;
        current_.q_vy = cfg_.mpc.q_vy;
        current_.q_r = cfg_.mpc.q_r;
        current_.r_steer = cfg_.mpc.r_steer;
        max_steps_ = static_cast<int>(2.5 * cfg_.distance_m /
                                    (budget_speed * cfg_.mpc.dt)) + 1000;
        trace_.open(dir_ / "bridge_trace.csv");
        if (!trace_) throw std::runtime_error("Cannot create bridge_trace.csv");
        trace_ << std::setprecision(17)
               << "step,x_m,y_m,vx_mps,ey_m,epsi_rad,steer_rad,np,q_y,q_psi,r_delta,qp_failed,grip_limited\n";
    }

    RlObservation observation() const {
        return makeRlObservation(state_, *reference_, current_.np, cfg_.distance_m,
            30.0, cfg_.rl, current_, gripAtCurrentX(), previous_steer_, grip_,
            plannedSteer(), cfg_.mpc.max_steer);
    }

    void sendReset() const {
        std::cout << "BRIDGE_RESET" << std::setprecision(17);
        for (double v : observation()) std::cout << ' ' << v;
        std::cout << std::endl;
    }

    void step(const std::array<double,4>& action) {
        if (ended_) throw std::logic_error("Episode ended; start a new bridge process");
        const MpcTuningAction previous_action = current_;
        current_.np = mappedNp(action[0], cfg_.rl, cfg_.mpc.nc);
        current_.q_y = mappedWeight(action[1], cfg_.rl.weight_min[0], cfg_.rl.weight_max[0]);
        current_.q_psi = mappedWeight(action[2], cfg_.rl.weight_min[1], cfg_.rl.weight_max[1]);
        current_.r_delta = mappedWeight(action[3], cfg_.rl.weight_min[5], cfg_.rl.weight_max[5]);
        // Qvy, Qr and Rsteer stay at the configured MPC values.
        // Keep exactly the same MPC object if the policy repeats its action.
        // Otherwise, transfer the previous MPC solution.
        const bool changed = !controller_ ||
            current_.np != previous_action.np ||
            current_.q_y != previous_action.q_y ||
            current_.q_psi != previous_action.q_psi ||
            current_.r_delta != previous_action.r_delta;
        if (changed) {
            const auto tuned = applyRlAction(cfg_.mpc, current_, cfg_.rl);
            auto updated = std::make_unique<LateralMpc>(vehicle_, tuned, *reference_, *tires_);
            if (controller_) updated->adoptPreviousSolution(*controller_);
            updated->setGripLimitedMode(grip_);
            controller_ = std::move(updated);
        }

        double reward_sum = 0.0;
        int steps = 0, qp_failed_count = 0, grip_count = 0;
        bool failed = false, terminated = false, truncated = false;
        double last_ey = 0.0, last_epsi = 0.0;
        for (int i = 0; i < cfg_.rl.decision_steps; ++i) {
            const double prior_steer = previous_steer_;
            vehicle_.road_grip_scale = gripAtCurrentX();
            try {
                controller_->setRoadGrip(vehicle_.road_grip_scale);
                MpcResult command;
                const auto before = state_;
                const auto start = std::chrono::steady_clock::now();
                try { command = controller_->control(state_, previous_steer_); }
                catch (const std::exception& e) {
                    command = controller_->enterGripLimitedMode(state_, previous_steer_);
                    command.prediction_failure = e.what();
                }
                grip_ = controller_->gripLimitedModeActive();
                const bool bad_qp = optimizationFailed(command);
                if (bad_qp) {
                    ++qp_failed_count;
                    if (cfg_.rl.penalize_qp_failure)
                        reward_sum -= cfg_.rl.reward.reward_scale * cfg_.rl.reward.failure_penalty;
                }
                if (!command.converged && !command.grip_limited)
                    throw std::runtime_error("MPC did not converge");
                if (command.grip_limited) ++grip_count;
                if (!std::isfinite(command.steer))
                    throw std::runtime_error("Nonfinite MPC steering");
                const auto elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                plant_->step(state_, command.steer, cfg_.mpc.dt);
                if (!std::isfinite(state_.x) || !std::isfinite(state_.y) ||
                    !std::isfinite(state_.vx) || std::abs(state_.y) > 20.0)
                    throw std::runtime_error("CarSim left numerical/safety range");
                previous_steer_ = command.steer;
                const auto next = reference_->at(state_.x);
                last_ey = state_.y - next.y;
                last_epsi = angleDifference(state_.psi, next.psi);
                SimulationRecord row;
                row.time = total_steps_ * cfg_.mpc.dt;
                row.s = before;
                row.reference = reference_->at(before.x);
                row.lateral_error = before.y - row.reference.y;
                row.heading_error = angleDifference(before.psi, row.reference.psi);
                row.steer = command.steer;
                row.yaw_rate_ref = before.vx * row.reference.curvature;
                row.road_grip = vehicle_.road_grip_scale;
                row.qp_iterations = command.iterations;
                row.qp_primal_residual = command.primal_residual;
                row.qp_dual_residual = command.dual_residual;
                row.qp_converged = command.converged;
                row.qp_solve_ms = elapsed;
                row.grip_limited_control = command.grip_limited;
                row.prediction_horizon = current_.np;
                row.q_y = current_.q_y; row.q_psi = current_.q_psi;
                row.q_vy = current_.q_vy; row.q_r = current_.q_r;
                row.r_steer = current_.r_steer; row.r_delta = current_.r_delta;
                // Match the original per-MPC-step reward; action-change penalty
                // is charged on the first step after a new policy action only.
                const auto reward_previous_action = i == 0 ? previous_action : current_;
                row.reward = scalarizedRlReward(last_ey, last_epsi, command.steer,
                    current_.np, cfg_.mpc, cfg_.rl, prior_steer,
                    i == 0 ? previous_action.np : current_.np,
                    current_, reward_previous_action, state_.vy,
                    state_.r - state_.vx * next.curvature, command.grip_limited);
                reward_sum += row.reward;
                if (bad_qp && cfg_.rl.penalize_qp_failure)
                    row.reward -= cfg_.rl.reward.reward_scale * cfg_.rl.reward.failure_penalty;
                records_.push_back(row);
                trace_ << total_steps_ << ',' << state_.x << ',' << state_.y << ','
                       << state_.vx << ',' << last_ey << ',' << last_epsi << ','
                       << command.steer << ',' << current_.np << ',' << current_.q_y
                       << ',' << current_.q_psi << ',' << current_.r_delta << ','
                       << (bad_qp ? 1 : 0) << ',' << (command.grip_limited ? 1 : 0) << '\n';
            } catch (const std::exception& e) {
                std::cerr << "BRIDGE_STEP_FAILURE " << e.what() << '\n';
                reward_sum -= cfg_.rl.reward.reward_scale * cfg_.rl.reward.failure_penalty;
                // Keep the last finite observation for failure transitions.
                if (std::isfinite(state_.x) && std::isfinite(state_.y) &&
                    std::isfinite(state_.psi)) {
                    const auto next = reference_->at(state_.x);
                    last_ey = state_.y - next.y;
                    last_epsi = angleDifference(state_.psi, next.psi);
                }
                failed = true;
            }
            ++steps;
            ++total_steps_;
            if (failed || state_.x >= cfg_.distance_m || total_steps_ >= max_steps_) {
                terminated = failed || state_.x >= cfg_.distance_m;
                truncated = !terminated && total_steps_ >= max_steps_;
                break;
            }
        }
        if (terminated || truncated) {
            ended_ = true;
            if (state_.x < cfg_.distance_m)
                reward_sum -= cfg_.rl.reward.reward_scale *
                              cfg_.rl.reward.incomplete_episode_penalty;
        }
        // One C++ reward formula is shared by all Python algorithms.
        const double mean = reward_sum / std::max(1, steps);
        const double reward = mean / cfg_.rl.reward.reward_scale;
        auto obs = observation();
        std::cout << "BRIDGE_STEP" << std::setprecision(17);
        for (double v : obs) std::cout << ' ' << v;
        std::cout << ' ' << reward << ' ' << (terminated ? 1 : 0)
                  << ' ' << (truncated ? 1 : 0) << ' ' << state_.x << ' '
                  << last_ey << ' ' << last_epsi << ' ' << previous_steer_
                  << ' ' << qp_failed_count << ' ' << grip_count << ' '
                  << steps << ' ' << (failed ? 1 : 0) << ' '
                  << current_.np << std::endl;
    }

    void exportReport(const std::string& label) {
        if (records_.empty()) throw std::runtime_error("No control steps to report");
        trace_.flush();
        RunOutcome outcome;
        outcome.adaptive = label != "neutral_mpc";
        outcome.requested_distance_m = cfg_.distance_m;
        outcome.completed = ended_ && state_.x >= cfg_.distance_m;
        outcome.stop_reason = outcome.completed ? "" : "Episode incomplete";
        outcome.reference_source = "CarSim LTARG_TABLE";
        outcome.carsim_info = readCarSimRunInfo(dir_ / "carsim_native" / "Run_all.par");
        outcome.variable_road_grip = model_.road_grip.variable();
        outcome.model_front_stiffness_n_per_rad = model_.front_stiffness_n_per_rad;
        outcome.model_rear_stiffness_n_per_rad = model_.rear_stiffness_n_per_rad;
        outcome.browser_result_status = "disabled for Python evaluation";
        auto snapshot = cfg_;
        snapshot.vehicle = vehicle_;
        snapshot.tire_data_file = "tire_data.csv";
        std::ofstream config_file(dir_ / "effective_config.ini");
        config_file << configToIni(snapshot);
        if (!config_file) throw std::runtime_error("Cannot write effective configuration");
        fs::copy_file(model_.tire_csv, dir_ / "tire_data.csv", fs::copy_options::overwrite_existing);
        exportResults(dir_, records_, vehicle_, cfg_.mpc, label, outcome);
    }
};
} // namespace

int main(int argc, char** argv) {
    // Pure numeric action-mapping smoke test, no CarSim DLL or Simfile needed.
    if (argc == 2 && std::string(argv[1]) == "--self-test-np") {
        try {
            RlConfig demo;
            demo.min_np = 18;
            demo.nominal_np = 25;
            demo.max_np = 40;
            const int min_horizon = mappedNp(-1.0, demo, 8);
            const int neutral_horizon = mappedNp(0.0, demo, 8);
            const int max_horizon = mappedNp(1.0, demo, 8);
            if (min_horizon != 18 || neutral_horizon != 25 || max_horizon != 40)
                throw std::runtime_error("Unexpected Np mapping");
            std::cout << "NP_MAPPING_PASS " << min_horizon << ' '
                      << neutral_horizon << ' ' << max_horizon << '\n';
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "NP_MAPPING_FAIL " << e.what() << '\n';
            return 1;
        }
    }
    // Use project-root-relative paths by launching from repository root.
    if (argc != 4) {
        std::cerr << "Usage: MPC_RL_Bridge <config.ini> <source.sim> <episode_output_dir>\n";
        return 2;
    }
    try {
        auto cfg = loadConfig(argv[1]);
        cfg.carsim.simfile = fs::absolute(argv[2]);
        cfg.open_report = false;
        validateConfig(cfg);
        std::unique_ptr<Episode> session;
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                std::istringstream in(line);
                std::string cmd, extra;
                in >> cmd;
                if (cmd == "RESET") {
                    if (session) throw std::runtime_error("One reset per process; restart for next episode");
                    if (in >> extra) throw std::runtime_error("RESET takes no arguments");
                    session = std::make_unique<Episode>(cfg, fs::absolute(argv[3]));
                    session->sendReset();
                } else if (cmd == "PARAMS") {
                    if (in >> extra) throw std::runtime_error("PARAMS takes no arguments");
                    const auto neutral = neutralRlAction(cfg.rl);
                    std::cout << "BRIDGE_PARAMS " << std::setprecision(17)
                              << neutral.np << ' ' << neutral.q_y << ' '
                              << neutral.q_psi << ' ' << cfg.mpc.q_vy << ' '
                              << cfg.mpc.q_r << ' ' << cfg.mpc.r_steer << ' '
                              << neutral.r_delta << ' ' << cfg.mpc.dt << ' '
                              << cfg.rl.decision_steps << ' ' << cfg.rl.discount_gamma
                              << ' ' << cfg.rl.reward.reward_scale << std::endl;
                } else if (cmd == "STEP") {
                    if (!session) throw std::runtime_error("STEP before RESET");
                    std::array<double,4> action{};
                    if (!(in >> action[0] >> action[1] >> action[2] >> action[3]) || (in >> extra))
                        throw std::runtime_error("STEP requires exactly four floats: np q_y q_psi r_delta");
                    for (double x : action)
                        if (!std::isfinite(x) || x < -1.0 || x > 1.0)
                            throw std::runtime_error("STEP actions must be finite values in [-1,1]");
                    session->step(action);
                } else if (cmd == "REPORT") {
                    if (!session) throw std::runtime_error("REPORT before RESET");
                    std::string label;
                    if (!(in >> label) || (in >> extra))
                        throw std::runtime_error("REPORT requires one label");
                    session->exportReport(label);
                    std::cout << "BRIDGE_REPORT" << std::endl;
                } else if (cmd == "CLOSE") {
                    std::cout << "BRIDGE_CLOSE" << std::endl;
                    break;
                } else throw std::runtime_error("Unknown bridge command");
            } catch (const std::exception& e) {
                std::cout << "BRIDGE_ERROR " << e.what() << std::endl;
                // The current episode might be inconsistent: terminate process.
                return 1;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cout << "BRIDGE_ERROR " << e.what() << std::endl;
        return 1;
    }
}
