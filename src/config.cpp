#include "config.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
constexpr double pi = 3.14159265358979323846264338327950288;
constexpr double deg2rad = pi / 180.0;
constexpr double kmh_per_mps = 3.6;

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
std::string lower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string canonicalConfigKey(std::string key) {
    if (key == "rl.train_speed_min_mps") return "rl.train_speed_min_kmh";
    if (key == "rl.train_speed_max_mps") return "rl.train_speed_max_kmh";
    return key;
}
double readDouble(const std::string& value, const std::string& key) {
    std::size_t used = 0;
    double v;
    try { v = std::stod(value, &used); }
    catch (const std::exception&) { throw std::invalid_argument("Invalid number for " + key + ": " + value); }
    if (used != value.size() || !std::isfinite(v))
        throw std::invalid_argument("Invalid finite number for " + key + ": " + value);
    return v;
}
int readInt(const std::string& value, const std::string& key) {
    std::size_t used = 0;
    int v;
    try { v = std::stoi(value, &used); }
    catch (const std::exception&) { throw std::invalid_argument("Invalid integer for " + key + ": " + value); }
    if (used != value.size())
        throw std::invalid_argument("Invalid integer for " + key + ": " + value);
    return v;
}
bool readBool(const std::string& value, const std::string& key) {
    const auto v = lower(value);
    if (v == "true" || v == "yes" || v == "1" || v == "on") return true;
    if (v == "false" || v == "no" || v == "0" || v == "off") return false;
    throw std::invalid_argument("Invalid boolean for " + key + ": " + value);
}
void positive(double n, const char* name) {
    if (!std::isfinite(n) || n <= 0.0)
        throw std::invalid_argument(std::string(name) + " must be positive and finite");
}
void nonnegative(double n, const char* name) {
    if (!std::isfinite(n) || n < 0.0)
        throw std::invalid_argument(std::string(name) + " must be nonnegative and finite");
}
}  // namespace

void overrideConfig(AppConfig& c, const std::string& qualified_key,
                    const std::string& raw_value) {
    const std::string key = lower(trim(qualified_key));
    const std::string value = trim(raw_value);
    if (key == "experiment.name") c.experiment = value;
    else if (key == "simulation.distance_m") c.distance_m = readDouble(value, key);
    else if (key == "simulation.output") c.output = value;
    else if (key == "simulation.open_report") c.open_report = readBool(value, key);
    else if (key == "simulation.plant") c.plant = lower(value);

    else if (key == "carsim.simfile") c.carsim.simfile = value;
    else if (key == "carsim.solver_dll") c.carsim.solver_dll = value;
    else if (key == "carsim.sync_browser_result") c.carsim.sync_browser_result = readBool(value, key);
    else if (key == "carsim.steering_left_index") c.carsim.steering_left_index = readInt(value, key);
    else if (key == "carsim.steering_right_index") c.carsim.steering_right_index = readInt(value, key);
    else if (key == "carsim.steering_gain") c.carsim.steering_gain = readDouble(value, key);
    else if (key == "carsim.x_keyword") c.carsim.state_keywords[0] = value;
    else if (key == "carsim.y_keyword") c.carsim.state_keywords[1] = value;
    else if (key == "carsim.yaw_keyword") c.carsim.state_keywords[2] = value;
    else if (key == "carsim.vx_keyword") c.carsim.state_keywords[3] = value;
    else if (key == "carsim.vy_keyword") c.carsim.state_keywords[4] = value;
    else if (key == "carsim.yaw_rate_keyword") c.carsim.state_keywords[5] = value;

    else if (key == "vehicle.mass_kg") c.vehicle.mass = readDouble(value, key);
    else if (key == "vehicle.iz_kgm2") c.vehicle.iz = readDouble(value, key);
    else if (key == "vehicle.a_m") c.vehicle.a = readDouble(value, key);
    else if (key == "vehicle.b_m") c.vehicle.b = readDouble(value, key);
    else if (key == "vehicle.speed_kmh") c.vehicle.vx = readDouble(value, key) / kmh_per_mps;
    else if (key == "tire.data_file") c.tire_data_file = value;
    else if (key == "tire.road_grip_scale") c.vehicle.road_grip_scale = readDouble(value, key);

    else if (key == "mpc.np") c.mpc.np = readInt(value, key);
    else if (key == "mpc.nc") c.mpc.nc = readInt(value, key);
    else if (key == "mpc.dt_s") c.mpc.dt = readDouble(value, key);
    else if (key == "mpc.speed_rate_enabled") c.mpc.speed_rate_enabled = readBool(value, key);
    else if (key == "mpc.speed_rate_start_kmh") c.mpc.speed_rate_start = readDouble(value, key) / kmh_per_mps;
    else if (key == "mpc.speed_rate_end_kmh") c.mpc.speed_rate_end = readDouble(value, key) / kmh_per_mps;
    else if (key == "mpc.speed_rate_end_deg_s") c.mpc.speed_rate_end_limit = readDouble(value, key) * deg2rad;
    else if (key == "mpc.grip_switch_enabled") c.mpc.grip_switch_enabled = readBool(value, key);
    else if (key == "mpc.grip_state_only") c.mpc.grip_state_only = readBool(value, key);
    else if (key == "mpc.grip_preview_s") c.mpc.grip_preview_s = readDouble(value, key);
    else if (key == "mpc.nonlinear_tire") c.mpc.nonlinear_tire = readBool(value, key);
    else if (key == "mpc.tire_slope_floor") c.mpc.tire_slope_floor = readDouble(value, key);

    else if (key == "weights.q_y") c.mpc.q_y = readDouble(value, key);
    else if (key == "weights.q_psi") c.mpc.q_psi = readDouble(value, key);
    else if (key == "weights.q_vy") c.mpc.q_vy = readDouble(value, key);
    else if (key == "weights.q_r") c.mpc.q_r = readDouble(value, key);
    else if (key == "weights.r_steer") c.mpc.r_steer = readDouble(value, key);
    else if (key == "weights.r_delta") c.mpc.r_delta = readDouble(value, key);
    else if (key == "weights.terminal_scale") c.mpc.terminal_scale = readDouble(value, key);

    else if (key == "constraints.max_steer_deg") c.mpc.max_steer = readDouble(value, key) * deg2rad;
    else if (key == "constraints.max_steer_rate_deg_s") c.mpc.max_steer_rate = readDouble(value, key) * deg2rad;

    else if (key == "solver.rho") c.mpc.admm_rho = readDouble(value, key);
    else if (key == "solver.max_iterations") c.mpc.admm_max_iterations = readInt(value, key);
    else if (key == "solver.primal_tol") c.mpc.admm_primal_tol = readDouble(value, key);
    else if (key == "solver.dual_tol") c.mpc.admm_dual_tol = readDouble(value, key);
    else if (key == "rl.mode") c.rl.mode = lower(value);
    else if (key == "rl.algorithm") c.rl.algorithm = lower(value);
    else if (key == "rl.action_profile") c.rl.action_profile = lower(value);
    else if (key == "rl.min_np") c.rl.min_np = readInt(value, key);
    else if (key == "rl.max_np") c.rl.max_np = readInt(value, key);
    else if (key == "rl.nominal_np") c.rl.nominal_np = readInt(value, key);
    else if (key == "rl.decision_steps") c.rl.decision_steps = readInt(value, key);
    else if (key == "rl.discount_gamma") c.rl.discount_gamma = readDouble(value, key);
    else if (key == "rl.actor_learning_rate") c.rl.actor_learning_rate = readDouble(value, key);
    else if (key == "rl.grip_demand_clip") c.rl.grip_demand_clip = readDouble(value, key);
    else if (key == "rl.np_exploration_scale") c.rl.np_exploration_scale = readDouble(value, key);
    else if (key == "rl.q_y_min") c.rl.weight_min[0] = readDouble(value, key);
    else if (key == "rl.q_y_max") c.rl.weight_max[0] = readDouble(value, key);
    else if (key == "rl.q_psi_min") c.rl.weight_min[1] = readDouble(value, key);
    else if (key == "rl.q_psi_max") c.rl.weight_max[1] = readDouble(value, key);
    else if (key == "rl.q_vy_min") c.rl.weight_min[2] = readDouble(value, key);
    else if (key == "rl.q_vy_max") c.rl.weight_max[2] = readDouble(value, key);
    else if (key == "rl.q_r_min") c.rl.weight_min[3] = readDouble(value, key);
    else if (key == "rl.q_r_max") c.rl.weight_max[3] = readDouble(value, key);
    else if (key == "rl.r_steer_min") c.rl.weight_min[4] = readDouble(value, key);
    else if (key == "rl.r_steer_max") c.rl.weight_max[4] = readDouble(value, key);
    else if (key == "rl.r_delta_min") c.rl.weight_min[5] = readDouble(value, key);
    else if (key == "rl.r_delta_max") c.rl.weight_max[5] = readDouble(value, key);
    else if (key == "rl.terminal_scale") c.rl.terminal_scale = readDouble(value, key);
    else if (key == "rl.curvature_preview_s") c.rl.curvature_preview_s = readDouble(value, key);
    else if (key == "rl.episodes") c.rl.episodes = readInt(value, key);
    else if (key == "rl.seed") c.rl.seed = readInt(value, key);
    else if (key == "rl.curriculum") c.rl.curriculum = readBool(value, key);
    else if (key == "rl.focus_target_speeds") c.rl.focus_target_speeds = readBool(value, key);
    else if (key == "rl.vary_road_transition") c.rl.vary_road_transition = readBool(value, key);
    else if (key == "rl.preserve_source_road") c.rl.preserve_source_road = readBool(value, key);
    else if (key == "rl.relative_advantage") c.rl.relative_advantage = readBool(value, key);
    else if (key == "rl.align_integer_np") c.rl.align_integer_np = readBool(value, key);
    else if (key == "rl.direct_block_reward") c.rl.direct_block_reward = readBool(value, key);
    else if (key == "rl.augmented_observation") c.rl.augmented_observation = readBool(value, key);
    else if (key == "rl.penalize_qp_failure") c.rl.penalize_qp_failure = readBool(value, key);
    else if (key == "rl.strict_tracking_selection") c.rl.strict_tracking_selection = readBool(value, key);
    else if (key == "rl.low_speed_blend") c.rl.low_speed_blend = readBool(value, key);
    else if (key == "rl.blend_start_kmh") c.rl.blend_start_kmh = readDouble(value, key);
    else if (key == "rl.blend_end_kmh") c.rl.blend_end_kmh = readDouble(value, key);
    else if (key == "rl.high_speed_blend") c.rl.high_speed_blend = readBool(value, key);
    else if (key == "rl.high_blend_start_kmh") c.rl.high_blend_start_kmh = readDouble(value, key);
    else if (key == "rl.high_blend_end_kmh") c.rl.high_blend_end_kmh = readDouble(value, key);
    else if (key == "rl.train_speed_min_kmh")
        c.rl.train_speed_min = readDouble(value, key) / kmh_per_mps;
    else if (key == "rl.train_speed_max_kmh")
        c.rl.train_speed_max = readDouble(value, key) / kmh_per_mps;
    else if (key == "rl.train_speed_min_mps") c.rl.train_speed_min = readDouble(value, key);
    else if (key == "rl.train_speed_max_mps") c.rl.train_speed_max = readDouble(value, key);
    else if (key == "rl.train_mu_min") c.rl.train_mu_min = readDouble(value, key);
    else if (key == "rl.train_mu_max") c.rl.train_mu_max = readDouble(value, key);
    else if (key == "rl.model_file") c.rl.model_file = value;
    else if (key == "rl.warm_start_file") c.rl.warm_start_file = value;
    else if (key == "rl.resume_state_file") c.rl.resume_state_file = value;
    else if (key == "rl.resume_actor_fallback") c.rl.resume_actor_fallback = readBool(value, key);
    else if (key == "rl_reward.reward_scale") c.rl.reward.reward_scale = readDouble(value, key);
    else if (key == "rl_reward.ey_scale_m") c.rl.reward.ey_scale_m = readDouble(value, key);
    else if (key == "rl_reward.epsi_scale_rad") c.rl.reward.epsi_scale_rad = readDouble(value, key);
    else if (key == "rl_reward.w_y") c.rl.reward.w_y = readDouble(value, key);
    else if (key == "rl_reward.w_peak") c.rl.reward.w_peak = readDouble(value, key);
    else if (key == "rl_reward.peak_threshold_m") c.rl.reward.peak_threshold_m = readDouble(value, key);
    else if (key == "rl_reward.w_psi") c.rl.reward.w_psi = readDouble(value, key);
    else if (key == "rl_reward.w_steer") c.rl.reward.w_steer = readDouble(value, key);
    else if (key == "rl_reward.w_steer_rate") c.rl.reward.w_steer_rate = readDouble(value, key);
    else if (key == "rl_reward.w_horizon") c.rl.reward.w_horizon = readDouble(value, key);
    else if (key == "rl_reward.w_np_change") c.rl.reward.w_np_change = readDouble(value, key);
    else if (key == "rl_reward.w_vy") c.rl.reward.w_vy = readDouble(value, key);
    else if (key == "rl_reward.w_yaw_error") c.rl.reward.w_yaw_error = readDouble(value, key);
    else if (key == "rl_reward.w_weight_change") c.rl.reward.w_weight_change = readDouble(value, key);
    else if (key == "rl_reward.w_weight_deviation") c.rl.reward.w_weight_deviation = readDouble(value, key);
    else if (key == "rl_reward.w_grip_limited") c.rl.reward.w_grip_limited = readDouble(value, key);
    else if (key == "rl_reward.failure_penalty") c.rl.reward.failure_penalty = readDouble(value, key);
    else if (key == "rl_reward.incomplete_episode_penalty")
        c.rl.reward.incomplete_episode_penalty = readDouble(value, key);
    else throw std::invalid_argument("Unknown configuration key: " + qualified_key);
}

AppConfig loadConfig(const std::filesystem::path& filename) {
    AppConfig result;
    result.source_config_file = std::filesystem::absolute(filename);
    std::set<std::filesystem::path> active_files;
    std::function<void(const std::filesystem::path&)> read_file;
    read_file = [&](const std::filesystem::path& requested) {
    const auto file = std::filesystem::absolute(requested).lexically_normal();
    if (!active_files.insert(file).second) throw std::runtime_error("Circular config inheritance: " + file.string());
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open config file: " + file.string());
    std::string line, section;
    std::set<std::string> keys;
    int line_number = 0;
    bool any_setting = false;
    while (std::getline(in, line)) {
        ++line_number;
        if (line_number == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xef &&
            static_cast<unsigned char>(line[1]) == 0xbb &&
            static_cast<unsigned char>(line[2]) == 0xbf)
            line.erase(0, 3); // UTF-8 BOM generated by some Windows editors.
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        try {
            if (line.front() == '[' && line.back() == ']') {
                section = lower(trim(line.substr(1, line.size() - 2)));
                if (section.empty()) throw std::invalid_argument("Empty section name");
                continue;
            }
            const auto eq = line.find('=');
            if (section.empty() || eq == std::string::npos)
                throw std::invalid_argument("Expected [section] followed by key=value");
            const std::string key = section + "." + lower(trim(line.substr(0, eq)));
            const std::string value = trim(line.substr(eq + 1));
            if (!keys.insert(canonicalConfigKey(key)).second)
                throw std::invalid_argument("Duplicate key: " + key);
            if (key == "inherit.base") {
                if (any_setting) throw std::invalid_argument("[inherit] base must precede all settings");
                const auto base = (file.parent_path() / value).lexically_normal();
                if (base.parent_path() != file.parent_path())
                    throw std::invalid_argument("Inherited config must be in the same directory");
                read_file(base);
                any_setting = true;
                continue;
            }
            overrideConfig(result, key, value);
            any_setting = true;
        } catch (const std::exception& ex) {
            throw std::runtime_error(file.string() + ":" +
                                     std::to_string(line_number) + ": " + ex.what());
        }
    }
    active_files.erase(file);
    };
    read_file(filename);
    if (!result.tire_data_file.empty() && result.tire_data_file.is_relative())
        result.tire_data_file = std::filesystem::absolute(filename).parent_path() /
                                result.tire_data_file;
    if (!result.carsim.simfile.empty() && result.carsim.simfile.is_relative())
        result.carsim.simfile = std::filesystem::absolute(filename).parent_path() /
                                result.carsim.simfile;
    if (!result.carsim.solver_dll.empty() && result.carsim.solver_dll.is_relative())
        result.carsim.solver_dll = std::filesystem::absolute(filename).parent_path() /
                                   result.carsim.solver_dll;
    if (!result.rl.warm_start_file.empty() && result.rl.warm_start_file.is_relative())
        result.rl.warm_start_file = std::filesystem::absolute(filename).parent_path() /
                                    result.rl.warm_start_file;
    if (!result.rl.resume_state_file.empty() && result.rl.resume_state_file.is_relative())
        result.rl.resume_state_file = std::filesystem::absolute(filename).parent_path() /
                                      result.rl.resume_state_file;
    if ((result.rl.mode == "eval" || result.rl.mode == "compare" || result.rl.mode == "compare_three" ||
         result.rl.mode == "eval_grid" || result.rl.mode == "eval_general" || result.rl.mode == "eval_focus") &&
        result.rl.model_file.is_relative())
        result.rl.model_file = std::filesystem::absolute(filename).parent_path() /
                               result.rl.model_file;
    validateConfig(result);
    return result;
}

void validateConfig(const AppConfig& c) {
    if (c.experiment.empty()) throw std::invalid_argument("experiment.name cannot be empty");
    if (c.output.empty()) throw std::invalid_argument("simulation.output cannot be empty");
    if (c.plant != "carsim")
        throw std::invalid_argument("simulation.plant must be carsim");
    {
        if (c.carsim.simfile.empty() || c.carsim.solver_dll.empty())
            throw std::invalid_argument("carsim.simfile and carsim.solver_dll are required");
        if (c.carsim.steering_left_index < 0 || c.carsim.steering_right_index < -1 ||
            c.carsim.steering_left_index == c.carsim.steering_right_index)
            throw std::invalid_argument("Invalid CarSim steering IMPORT indices");
        positive(c.carsim.steering_gain, "carsim.steering_gain");
        for (const auto& keyword : c.carsim.state_keywords)
            if (keyword.empty()) throw std::invalid_argument("CarSim state keyword cannot be empty");
    }
    positive(c.distance_m, "simulation.distance_m");
    if (c.distance_m > 100000.0) throw std::invalid_argument("Distance is too large");
    const auto& v = c.vehicle;
    positive(v.mass, "mass_kg"); positive(v.iz, "iz_kgm2");
    positive(v.a, "a_m"); positive(v.b, "b_m");
    positive(v.vx, "speed_kmh");
    if (v.vx < 1.0) throw std::invalid_argument("speed_kmh must be >=3.6 for this model");
    positive(v.road_grip_scale, "road_grip_scale");
    if (v.road_grip_scale > 2.0)
        throw std::invalid_argument("road_grip_scale must be <= 2; higher grip needs measured data");
    const auto& p = c.mpc;
    if (p.np < 2 || p.np > 150 || p.nc < 1 || p.nc > p.np)
        throw std::invalid_argument("MPC horizons must satisfy 1 <= Nc <= Np <= 150 and Np >= 2");
    positive(p.dt, "mpc.dt_s");
    nonnegative(p.speed_rate_start, "mpc.speed_rate_start_kmh");
    positive(p.speed_rate_end, "mpc.speed_rate_end_kmh");
    positive(p.speed_rate_end_limit, "mpc.speed_rate_end_deg_s");
    if (p.speed_rate_end <= p.speed_rate_start)
        throw std::invalid_argument("mpc.speed_rate_end_kmh must exceed speed_rate_start_kmh");
    if (p.dt < 0.005 || p.dt > 0.15)
        throw std::invalid_argument("mpc.dt_s must be between 0.005 and 0.15");
    nonnegative(p.q_y, "q_y"); nonnegative(p.q_psi, "q_psi");
    nonnegative(p.q_vy, "q_vy"); nonnegative(p.q_r, "q_r");
    nonnegative(p.r_steer, "r_steer"); nonnegative(p.r_delta, "r_delta");
    positive(p.terminal_scale, "terminal_scale");
    if (p.q_y+p.q_psi+p.q_vy+p.q_r+p.r_steer+p.r_delta <= 0.0)
        throw std::invalid_argument("At least one MPC cost weight must be positive");
    positive(p.max_steer, "max_steer_deg"); positive(p.max_steer_rate, "max_steer_rate_deg_s");
    positive(p.admm_rho, "solver.rho");
    positive(p.admm_primal_tol, "solver.primal_tol");
    positive(p.admm_dual_tol, "solver.dual_tol");
    if (p.admm_max_iterations < 10 || p.admm_max_iterations > 100000)
        throw std::invalid_argument("solver.max_iterations must be 10..100000");
    const auto& rl = c.rl;
    nonnegative(rl.blend_start_kmh, "rl.blend_start_kmh");
    positive(rl.blend_end_kmh, "rl.blend_end_kmh");
    if (rl.blend_end_kmh <= rl.blend_start_kmh)
        throw std::invalid_argument("rl.blend_end_kmh must exceed blend_start_kmh");
    nonnegative(rl.high_blend_start_kmh, "rl.high_blend_start_kmh");
    positive(rl.high_blend_end_kmh, "rl.high_blend_end_kmh");
    if (rl.high_blend_end_kmh <= rl.high_blend_start_kmh)
        throw std::invalid_argument("rl.high_blend_end_kmh must exceed high_blend_start_kmh");
    if (rl.mode != "fixed" && rl.mode != "train" && rl.mode != "eval" &&
        rl.mode != "compare" && rl.mode != "compare_three" && rl.mode != "eval_grid" && rl.mode != "eval_general" && rl.mode != "eval_focus" && rl.mode != "export_state")
        throw std::invalid_argument("Invalid RL mode");
    for (std::size_t k = 0; k < rl.weight_min.size(); ++k)
        if (!(std::isfinite(rl.weight_min[k]) && std::isfinite(rl.weight_max[k]) &&
              rl.weight_min[k] > 0.0 && rl.weight_max[k] > rl.weight_min[k] &&
              rl.weight_max[k] / rl.weight_min[k] <= 100.0))
            throw std::invalid_argument("RL absolute Q/R bounds must be positive, increasing, and have ratio <= 100");
    positive(rl.terminal_scale, "rl.terminal_scale");
    if (rl.algorithm != "ddpg" && rl.algorithm != "td3")
        throw std::invalid_argument("rl.algorithm must be ddpg or td3");
    if (rl.action_profile != "full" && rl.action_profile != "compact" && rl.action_profile != "np_only")
        throw std::invalid_argument("rl.action_profile must be full, compact or np_only");
    if (rl.augmented_observation && rl.action_profile != "compact")
        throw std::invalid_argument("augmented observation requires compact action profile");
    if (rl.min_np < 2 || (rl.mode != "fixed" && rl.min_np < p.nc) ||
        rl.max_np > 150 || rl.max_np <= rl.min_np)
        throw std::invalid_argument("RL horizons require Nc <= min_np < max_np <= 150");
    if (rl.nominal_np <= rl.min_np || rl.nominal_np >= rl.max_np)
        throw std::invalid_argument("rl.nominal_np must lie strictly inside RL min_np..max_np");
    if (rl.decision_steps < 1 || rl.decision_steps > 100 || rl.episodes < 1 || rl.episodes > 10000)
        throw std::invalid_argument("Invalid RL decision_steps or episodes");
    if (!(rl.discount_gamma > 0.0 && rl.discount_gamma <= 1.0))
        throw std::invalid_argument("rl.discount_gamma must be in (0, 1]");
    positive(rl.np_exploration_scale, "rl.np_exploration_scale");
    positive(rl.actor_learning_rate, "rl.actor_learning_rate");
    positive(rl.grip_demand_clip, "rl.grip_demand_clip");
    if (rl.grip_demand_clip != 3.0 && rl.grip_demand_clip != 4.0)
        throw std::invalid_argument("rl.grip_demand_clip must be 3 or 4");
    if (rl.grip_demand_clip != 3.0 && (rl.action_profile != "compact" || rl.augmented_observation))
        throw std::invalid_argument("non-default grip demand observation requires non-augmented compact policy");
    if (rl.seed < 0 || rl.train_speed_min < 1.0 ||
        rl.train_speed_max < rl.train_speed_min || rl.train_speed_max > 100.0)
        throw std::invalid_argument("Invalid RL seed or training speed range");
    if (!std::isfinite(rl.train_mu_min) || !std::isfinite(rl.train_mu_max) ||
        rl.train_mu_min < 0.1 || rl.train_mu_max > 1.5 ||
        rl.train_mu_min > rl.train_mu_max)
        throw std::invalid_argument("RL training road friction range must satisfy 0.1 <= min <= max <= 1.5");
    if (rl.model_file.empty()) throw std::invalid_argument("rl.model_file cannot be empty");
    if (!rl.warm_start_file.empty() && !rl.resume_state_file.empty())
        throw std::invalid_argument("Choose only one of rl.warm_start_file and rl.resume_state_file");
    positive(rl.curvature_preview_s, "rl.curvature_preview_s");
    if (rl.curvature_preview_s > 10.0)
        throw std::invalid_argument("rl.curvature_preview_s must be <= 10 s");
    positive(rl.reward.reward_scale, "rl_reward.reward_scale");
    positive(rl.reward.ey_scale_m, "rl_reward.ey_scale_m");
    positive(rl.reward.epsi_scale_rad, "rl_reward.epsi_scale_rad");
    nonnegative(rl.reward.w_y, "rl_reward.w_y");
    nonnegative(rl.reward.w_peak, "rl_reward.w_peak");
    positive(rl.reward.peak_threshold_m, "rl_reward.peak_threshold_m");
    nonnegative(rl.reward.w_psi, "rl_reward.w_psi");
    nonnegative(rl.reward.w_steer, "rl_reward.w_steer");
    nonnegative(rl.reward.w_steer_rate, "rl_reward.w_steer_rate");
    nonnegative(rl.reward.w_horizon, "rl_reward.w_horizon");
    nonnegative(rl.reward.w_np_change, "rl_reward.w_np_change");
    nonnegative(rl.reward.w_vy, "rl_reward.w_vy");
    nonnegative(rl.reward.w_yaw_error, "rl_reward.w_yaw_error");
    nonnegative(rl.reward.w_weight_change, "rl_reward.w_weight_change");
    nonnegative(rl.reward.w_weight_deviation, "rl_reward.w_weight_deviation");
    nonnegative(rl.reward.w_grip_limited, "rl_reward.w_grip_limited");
    positive(rl.reward.failure_penalty, "rl_reward.failure_penalty");
    nonnegative(rl.reward.incomplete_episode_penalty,
                "rl_reward.incomplete_episode_penalty");
    if (rl.reward.w_y + rl.reward.w_psi + rl.reward.w_steer +
        rl.reward.w_steer_rate + rl.reward.w_horizon +
        rl.reward.w_np_change <= 0.0)
        throw std::invalid_argument("At least one RL reward weight must be positive");
}

std::string configToIni(const AppConfig& c) {
    std::ostringstream out;
    out << std::setprecision(17) << std::boolalpha;
    out << "# Effective controller inputs and observed CarSim model parameters.\n";
    out << "# Vehicle and tire values below are read from this run's CarSim solver echo.\n"
           "# speed_kmh is the observed initial speed, not a speed command.\n"
           "# CarSim vehicle/tire source: carsim_native/LastRun_echo.par\n"
           "# CarSim road, speed and path source: carsim_native/Run_all.par\n";
    out
        << "[experiment]\nname = " << c.experiment << "\n\n"
        << "[simulation]\ndistance_m = " << c.distance_m
        << "\noutput = " << c.output.generic_string()
        << "\nopen_report = " << c.open_report
        << "\nplant = " << c.plant << "\n\n"
        << "[carsim]\nsimfile = " << c.carsim.simfile.generic_string()
        << "\nsolver_dll = " << c.carsim.solver_dll.generic_string()
        << "\nsync_browser_result = " << c.carsim.sync_browser_result
        << "\nsteering_left_index = " << c.carsim.steering_left_index
        << "\nsteering_right_index = " << c.carsim.steering_right_index
        << "\nsteering_gain = " << c.carsim.steering_gain
        << "\nx_keyword = " << c.carsim.state_keywords[0]
        << "\ny_keyword = " << c.carsim.state_keywords[1]
        << "\nyaw_keyword = " << c.carsim.state_keywords[2]
        << "\nvx_keyword = " << c.carsim.state_keywords[3]
        << "\nvy_keyword = " << c.carsim.state_keywords[4]
        << "\nyaw_rate_keyword = " << c.carsim.state_keywords[5] << "\n\n"
        << "[vehicle]\nmass_kg = " << c.vehicle.mass
        << "\niz_kgm2 = " << c.vehicle.iz
        << "\na_m = " << c.vehicle.a
        << "\nb_m = " << c.vehicle.b
        << std::setprecision(15)
        << "\nspeed_kmh = " << c.vehicle.vx * kmh_per_mps
        << std::setprecision(17)
        << "\n\n"
        << "[tire]\ndata_file = " << c.tire_data_file.generic_string()
        << "\nroad_grip_scale = " << c.vehicle.road_grip_scale << "\n\n"
        << "[mpc]\nnp = " << c.mpc.np
        << "\nnc = " << c.mpc.nc
        << "\ndt_s = " << c.mpc.dt
        << "\nspeed_rate_enabled = " << c.mpc.speed_rate_enabled
        << "\nspeed_rate_start_kmh = " << c.mpc.speed_rate_start * kmh_per_mps
        << "\nspeed_rate_end_kmh = " << c.mpc.speed_rate_end * kmh_per_mps
        << "\nspeed_rate_end_deg_s = " << c.mpc.speed_rate_end_limit / deg2rad
        << "\ngrip_switch_enabled = " << c.mpc.grip_switch_enabled
        << "\ngrip_state_only = " << c.mpc.grip_state_only
        << "\ngrip_preview_s = " << c.mpc.grip_preview_s
        << "\nnonlinear_tire = " << c.mpc.nonlinear_tire
        << "\ntire_slope_floor = " << c.mpc.tire_slope_floor << "\n\n"
        << "[weights]\nq_y = " << c.mpc.q_y
        << "\nq_psi = " << c.mpc.q_psi
        << "\nq_vy = " << c.mpc.q_vy
        << "\nq_r = " << c.mpc.q_r
        << "\nr_steer = " << c.mpc.r_steer
        << "\nr_delta = " << c.mpc.r_delta
        << "\nterminal_scale = " << c.mpc.terminal_scale << "\n\n"
        << "[constraints]\nmax_steer_deg = " << c.mpc.max_steer / deg2rad
        << "\nmax_steer_rate_deg_s = " << c.mpc.max_steer_rate / deg2rad << "\n\n"
        << "[solver]\nrho = " << c.mpc.admm_rho
        << "\nmax_iterations = " << c.mpc.admm_max_iterations
        << "\nprimal_tol = " << c.mpc.admm_primal_tol
        << "\ndual_tol = " << c.mpc.admm_dual_tol << '\n';
    out << "\n[rl]\nmode = " << c.rl.mode
        << "\nalgorithm = " << c.rl.algorithm
        << "\naction_profile = " << c.rl.action_profile
        << "\nmin_np = " << c.rl.min_np
        << "\nmax_np = " << c.rl.max_np
        << "\nnominal_np = " << c.rl.nominal_np
        << "\ndecision_steps = " << c.rl.decision_steps
        << "\ndiscount_gamma = " << c.rl.discount_gamma
        << "\nnp_exploration_scale = " << c.rl.np_exploration_scale
        << "\nq_y_min = " << c.rl.weight_min[0]
        << "\nq_y_max = " << c.rl.weight_max[0]
        << "\nq_psi_min = " << c.rl.weight_min[1]
        << "\nq_psi_max = " << c.rl.weight_max[1]
        << "\nq_vy_min = " << c.rl.weight_min[2]
        << "\nq_vy_max = " << c.rl.weight_max[2]
        << "\nq_r_min = " << c.rl.weight_min[3]
        << "\nq_r_max = " << c.rl.weight_max[3]
        << "\nr_steer_min = " << c.rl.weight_min[4]
        << "\nr_steer_max = " << c.rl.weight_max[4]
        << "\nr_delta_min = " << c.rl.weight_min[5]
        << "\nr_delta_max = " << c.rl.weight_max[5]
        << "\nterminal_scale = " << c.rl.terminal_scale
        << "\ncurvature_preview_s = " << c.rl.curvature_preview_s
        << "\nepisodes = " << c.rl.episodes
        << "\nseed = " << c.rl.seed
        << "\ncurriculum = " << c.rl.curriculum
        << "\nfocus_target_speeds = " << c.rl.focus_target_speeds
        << "\nvary_road_transition = " << c.rl.vary_road_transition
        << "\npreserve_source_road = " << c.rl.preserve_source_road
        << "\nrelative_advantage = " << c.rl.relative_advantage
        << "\nactor_learning_rate = " << c.rl.actor_learning_rate
        << "\ngrip_demand_clip = " << c.rl.grip_demand_clip
        << "\nalign_integer_np = " << c.rl.align_integer_np
        << "\ndirect_block_reward = " << c.rl.direct_block_reward
        << "\naugmented_observation = " << c.rl.augmented_observation
        << "\npenalize_qp_failure = " << c.rl.penalize_qp_failure
        << "\nstrict_tracking_selection = " << c.rl.strict_tracking_selection
        << "\nlow_speed_blend = " << c.rl.low_speed_blend
        << "\nblend_start_kmh = " << c.rl.blend_start_kmh
        << "\nblend_end_kmh = " << c.rl.blend_end_kmh
        << "\nhigh_speed_blend = " << c.rl.high_speed_blend
        << "\nhigh_blend_start_kmh = " << c.rl.high_blend_start_kmh
        << "\nhigh_blend_end_kmh = " << c.rl.high_blend_end_kmh
        << std::setprecision(15)
        << "\ntrain_speed_min_kmh = " << c.rl.train_speed_min * kmh_per_mps
        << "\ntrain_speed_max_kmh = " << c.rl.train_speed_max * kmh_per_mps
        << "\ntrain_mu_min = " << c.rl.train_mu_min
        << "\ntrain_mu_max = " << c.rl.train_mu_max
        << std::setprecision(17)
        << "\nmodel_file = " << c.rl.model_file.generic_string()
        << "\nwarm_start_file = " << c.rl.warm_start_file.generic_string()
        << "\nresume_state_file = " << c.rl.resume_state_file.generic_string()
        << "\nresume_actor_fallback = " << c.rl.resume_actor_fallback << "\n\n"
        << "[rl_reward]\nreward_scale = " << c.rl.reward.reward_scale
        << "\ney_scale_m = " << c.rl.reward.ey_scale_m
        << "\nepsi_scale_rad = " << c.rl.reward.epsi_scale_rad
        << "\nw_y = " << c.rl.reward.w_y
        << "\nw_peak = " << c.rl.reward.w_peak
        << "\npeak_threshold_m = " << c.rl.reward.peak_threshold_m
        << "\nw_psi = " << c.rl.reward.w_psi
        << "\nw_steer = " << c.rl.reward.w_steer
        << "\nw_steer_rate = " << c.rl.reward.w_steer_rate
        << "\nw_horizon = " << c.rl.reward.w_horizon
        << "\nw_np_change = " << c.rl.reward.w_np_change
        << "\nw_vy = " << c.rl.reward.w_vy
        << "\nw_yaw_error = " << c.rl.reward.w_yaw_error
        << "\nw_weight_change = " << c.rl.reward.w_weight_change
        << "\nw_weight_deviation = " << c.rl.reward.w_weight_deviation
        << "\nw_grip_limited = " << c.rl.reward.w_grip_limited
        << "\nfailure_penalty = " << c.rl.reward.failure_penalty
        << "\nincomplete_episode_penalty = "
        << c.rl.reward.incomplete_episode_penalty << '\n';
    return out.str();
}

std::filesystem::path locateDefaultConfig(const std::filesystem::path& executable) {
    auto checkAncestors = [](std::filesystem::path p) -> std::filesystem::path {
        for (int i = 0; i < 7; ++i) {
            auto candidate = p / "configs" / "default.ini";
            if (std::filesystem::is_regular_file(candidate))
                return std::filesystem::absolute(candidate);
            if (p == p.parent_path() || p.empty()) break;
            p = p.parent_path();
        }
        return {};
    };
    auto from_cwd = checkAncestors(std::filesystem::current_path());
    if (!from_cwd.empty()) return from_cwd;
    auto from_exe = checkAncestors(std::filesystem::absolute(executable).parent_path());
    if (!from_exe.empty()) return from_exe;
    throw std::runtime_error(
        "Cannot find configs/default.ini. Run from project folder or pass --config PATH");
}
