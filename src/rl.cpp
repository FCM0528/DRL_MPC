#include "rl.h"
#include "config.h"
#include "visualization.h"
#include "carsim_plant.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <set>
#include <stdexcept>
#include <tuple>

namespace {
double clip(double x, double lo, double hi) { return std::clamp(x, lo, hi); }

// Small fully connected network. Hidden activation is tanh; the actor output
// is tanh and the critic output is linear. Adam state belongs to each layer.
class Network {
    struct Layer {
        int in, out;
        std::vector<double> w, b, mw, vw, mb, vb;
        Layer(int input, int output, std::mt19937& rng) : in(input), out(output),
            w(static_cast<std::size_t>(input * output)), b(output, 0.0),
            mw(w.size(), 0.0), vw(w.size(), 0.0), mb(output, 0.0), vb(output, 0.0) {
            std::normal_distribution<double> init(0.0, std::sqrt(1.0 / input));
            for (double& value : w) value = init(rng);
        }
    };
    std::vector<Layer> layers_;
    bool bounded_output_;
    unsigned long long updates_ = 0;
public:
    Network(int input, int hidden, int output, bool bounded, std::mt19937& rng)
        : bounded_output_(bounded) {
        layers_.emplace_back(input, hidden, rng);
        layers_.emplace_back(hidden, hidden, rng);
        layers_.emplace_back(hidden, output, rng);
        // Zero logits start at the RL action center, independent of fixed MPC.
        if (bounded) {
            std::fill(layers_.back().w.begin(), layers_.back().w.end(), 0.0);
            std::fill(layers_.back().b.begin(), layers_.back().b.end(), 0.0);
        }
    }
    std::vector<double> forward(const std::vector<double>& input) const {
        auto values = input;
        for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
            const auto& l = layers_[layer];
            std::vector<double> next(l.out);
            for (int j = 0; j < l.out; ++j) {
                double sum = l.b[j];
                for (int i = 0; i < l.in; ++i) sum += l.w[j * l.in + i] * values[i];
                next[j] = (layer + 1 == layers_.size() && !bounded_output_)
                    ? sum : std::tanh(sum);
            }
            values = std::move(next);
        }
        return values;
    }
    double value(const std::vector<double>& input) const { return forward(input)[0]; }
    std::vector<double> backward(const std::vector<double>& input,
                                 std::vector<double> gradient, double learning_rate) {
        std::vector<std::vector<double>> activations{input};
        for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
            const auto& l = layers_[layer];
            std::vector<double> next(l.out);
            for (int j = 0; j < l.out; ++j) {
                double sum = l.b[j];
                for (int i = 0; i < l.in; ++i)
                    sum += l.w[j * l.in + i] * activations.back()[i];
                next[j] = (layer + 1 == layers_.size() && !bounded_output_)
                    ? sum : std::tanh(sum);
            }
            activations.push_back(std::move(next));
        }
        if (gradient.size() != activations.back().size())
            throw std::invalid_argument("Network gradient dimension mismatch");
        if (learning_rate > 0.0) ++updates_;
        for (int layer = static_cast<int>(layers_.size()) - 1; layer >= 0; --layer) {
            auto& l = layers_[static_cast<std::size_t>(layer)];
            if (layer + 1 != static_cast<int>(layers_.size()) || bounded_output_)
                for (int j = 0; j < l.out; ++j)
                    gradient[j] *= 1.0 - activations[layer + 1][j] * activations[layer + 1][j];
            std::vector<double> previous(l.in, 0.0);
            for (int j = 0; j < l.out; ++j)
                for (int i = 0; i < l.in; ++i)
                    previous[i] += l.w[j * l.in + i] * gradient[j];
            if (learning_rate > 0.0) {
                const double correction1 = 1.0 - std::pow(0.9, static_cast<double>(updates_));
                const double correction2 = 1.0 - std::pow(0.999, static_cast<double>(updates_));
                auto adam = [&](double& parameter, double& m, double& v, double derivative) {
                    derivative = clip(derivative, -10.0, 10.0);
                    m = 0.9 * m + 0.1 * derivative;
                    v = 0.999 * v + 0.001 * derivative * derivative;
                    parameter -= learning_rate * (m / correction1) /
                                 (std::sqrt(v / correction2) + 1e-8);
                };
                for (int j = 0; j < l.out; ++j) {
                    for (int i = 0; i < l.in; ++i) {
                        const auto k = static_cast<std::size_t>(j * l.in + i);
                        adam(l.w[k], l.mw[k], l.vw[k],
                             gradient[j] * activations[layer][i]);
                    }
                    adam(l.b[j], l.mb[j], l.vb[j], gradient[j]);
                }
            }
            gradient = std::move(previous);
        }
        return gradient;
    }
    std::vector<double> backward(const std::vector<double>& input,
                                 double output_gradient, double learning_rate) {
        return backward(input, std::vector<double>{output_gradient}, learning_rate);
    }
    void softUpdate(const Network& source, double tau) {
        for (std::size_t k = 0; k < layers_.size(); ++k) {
            auto& dst = layers_[k];
            const auto& src = source.layers_[k];
            for (std::size_t i = 0; i < dst.w.size(); ++i)
                dst.w[i] = (1.0 - tau) * dst.w[i] + tau * src.w[i];
            for (std::size_t i = 0; i < dst.b.size(); ++i)
                dst.b[i] = (1.0 - tau) * dst.b[i] + tau * src.b[i];
        }
    }
    void save(std::ostream& out) const {
        for (const auto& l : layers_) {
            for (double x : l.w) out << x << ' ';
            for (double x : l.b) out << x << ' ';
            out << '\n';
        }
    }
    void load(std::istream& in) {
        for (auto& l : layers_) {
            for (double& x : l.w) if (!(in >> x) || !std::isfinite(x))
                throw std::runtime_error("Invalid policy weights");
            for (double& x : l.b) if (!(in >> x) || !std::isfinite(x))
                throw std::runtime_error("Invalid policy bias");
        }
    }
};

std::vector<double> vectorOf(const RlObservation& observation) {
    return {observation.begin(), observation.end()};
}
using ContinuousAction = std::array<double, 7>;
std::vector<double> criticInput(const RlObservation& state,
                                const ContinuousAction& action) {
    auto input = vectorOf(state);
    input.insert(input.end(), action.begin(), action.end());
    return input;
}
struct Transition {
    RlObservation state, next;
    ContinuousAction action;
    double reward;
    bool done;
};

class DdpgPolicy final : public RlPolicy {
    RlConfig config_;
    int nominal_np_;
    std::mt19937 rng_;
    Network actor_, actor_target_, critic_, critic_target_;
    std::vector<Transition> replay_;
    std::size_t replay_position_ = 0;
    std::normal_distribution<double> noise_{0.0, 1.0};
    double exploration_sigma_ = 0.25;
    static double actionTrust(const RlObservation& observation) {
        // Keep the actor active at every speed while limiting large tuning
        // jumps where the low-speed kinematic baseline already tracks closely.
        const double speed_kmh = 108.0 * observation[0];
        const double t = clip((speed_kmh - 30.0) / 45.0, 0.0, 1.0);
        const double base = 0.02 + 0.98 * t * t * (3.0 - 2.0 * t);
        const double transition = (speed_kmh - 36.0) / 2.0;
        return base * (1.0 - 0.5 * std::exp(-0.5 * transition * transition));
    }
    double encodeNp(int np) const {
        return np >= nominal_np_
            ? static_cast<double>(np - nominal_np_) / (config_.max_np - nominal_np_)
            : static_cast<double>(np - nominal_np_) / (nominal_np_ - config_.min_np);
    }
    int decodeNp(double action) const {
        return std::clamp(static_cast<int>(std::lround(nominal_np_ +
            clip(action, -1.0, 1.0) * (action >= 0.0
                ? config_.max_np - nominal_np_ : nominal_np_ - config_.min_np))),
            config_.min_np, config_.max_np);
    }
    static double decodeWeight(double value, double minimum, double maximum) {
        return std::exp(0.5 * ((1.0 - clip(value, -1.0, 1.0)) * std::log(minimum) +
                               (1.0 + clip(value, -1.0, 1.0)) * std::log(maximum)));
    }
    static double encodeWeight(double value, double minimum, double maximum) {
        return clip(2.0 * (std::log(value) - std::log(minimum)) /
            (std::log(maximum) - std::log(minimum)) - 1.0, -1.0, 1.0);
    }
    ContinuousAction encode(const MpcTuningAction& action) const {
        return {encodeNp(action.np),
            encodeWeight(action.q_y, config_.weight_min[0], config_.weight_max[0]),
            encodeWeight(action.q_psi, config_.weight_min[1], config_.weight_max[1]),
            encodeWeight(action.q_vy, config_.weight_min[2], config_.weight_max[2]),
            encodeWeight(action.q_r, config_.weight_min[3], config_.weight_max[3]),
            encodeWeight(action.r_steer, config_.weight_min[4], config_.weight_max[4]),
            encodeWeight(action.r_delta, config_.weight_min[5], config_.weight_max[5])};
    }
    MpcTuningAction decode(const ContinuousAction& values) const {
        MpcTuningAction action;
        action.np = decodeNp(values[0]);
        action.q_y = decodeWeight(values[1], config_.weight_min[0], config_.weight_max[0]);
        action.q_psi = decodeWeight(values[2], config_.weight_min[1], config_.weight_max[1]);
        action.q_vy = decodeWeight(values[3], config_.weight_min[2], config_.weight_max[2]);
        action.q_r = decodeWeight(values[4], config_.weight_min[3], config_.weight_max[3]);
        action.r_steer = decodeWeight(values[5], config_.weight_min[4], config_.weight_max[4]);
        action.r_delta = decodeWeight(values[6], config_.weight_min[5], config_.weight_max[5]);
        return action;
    }
    void update() {
        if (replay_.size() < 64) return;
        std::uniform_int_distribution<std::size_t> sample(0, replay_.size() - 1);
        for (int i = 0; i < 32; ++i) {
            const auto& tr = replay_[sample(rng_)];
            const auto next_vector = actor_target_.forward(vectorOf(tr.next));
            ContinuousAction next_action{};
            for (int k = 0; k < 7; ++k)
                next_action[k] = next_vector[k] * actionTrust(tr.next);
            const double target = tr.reward + (tr.done ? 0.0 :
                0.98 * critic_target_.value(criticInput(tr.next, next_action)));
            const auto input = criticInput(tr.state, tr.action);
            const double prediction = critic_.value(input);
            critic_.backward(input, clip(prediction - target, -10.0, 10.0), 0.0003);
        }
        // Deterministic policy gradient through the critic's action input.
        for (int i = 0; i < 4; ++i) {
            const auto& tr = replay_[sample(rng_)];
            const auto state = vectorOf(tr.state);
            const auto output = actor_.forward(state);
            ContinuousAction action{};
            const double trust = actionTrust(tr.state);
            for (int k = 0; k < 7; ++k) action[k] = output[k] * trust;
            const auto input_gradient = critic_.backward(
                criticInput(tr.state, action), 1.0, 0.0);
            std::vector<double> actor_gradient(7);
            for (int k = 0; k < 7; ++k)
                actor_gradient[k] = -clip(input_gradient[17 + k] * trust, -10.0, 10.0);
            actor_.backward(state, actor_gradient, 0.00005);
        }
        actor_target_.softUpdate(actor_, 0.005);
        critic_target_.softUpdate(critic_, 0.005);
    }
public:
    explicit DdpgPolicy(const RlConfig& config)
        : config_(config), nominal_np_(config.nominal_np),
        rng_(static_cast<std::mt19937::result_type>(config.seed)),
        actor_(17, 64, 7, true, rng_), actor_target_(actor_),
        critic_(24, 64, 1, false, rng_), critic_target_(critic_) {
        if (nominal_np_ <= config_.min_np || nominal_np_ >= config_.max_np)
            throw std::invalid_argument("DDPG nominal Np must lie inside action bounds");
        replay_.reserve(20000);
    }
    MpcTuningAction act(const RlObservation& observation, bool explore) override {
        auto values = actor_.forward(vectorOf(observation));
        ContinuousAction action{};
        for (int k = 0; k < 7; ++k)
            action[k] = clip((values[k] + (explore ? exploration_sigma_ * noise_(rng_) : 0.0)) *
                             actionTrust(observation), -1.0, 1.0);
        return decode(action);
    }
    void setTrainingEpisode(int episode, int total) override {
        const double progress = total <= 1 ? 1.0 :
            clip(static_cast<double>(episode - 1) / (total - 1), 0.0, 1.0);
        exploration_sigma_ = 0.04 + 0.21 * (1.0 - progress) * (1.0 - progress);
    }
    double explorationStdDev() const override { return exploration_sigma_; }
    void observe(const RlObservation& state, const MpcTuningAction& action,
                 double reward, const RlObservation& next, bool done) override {
        const Transition tr{state, next, encode(action), reward, done};
        if (replay_.size() < 20000) replay_.push_back(tr);
        else { replay_[replay_position_] = tr; replay_position_ = (replay_position_ + 1) % 20000; }
        update();
    }
    void save(const std::filesystem::path& file) const override {
        std::ofstream out(file, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write DDPG policy: " + file.string());
        out << std::setprecision(17) << "DDPG_MPC_V15_ABSOLUTE " << nominal_np_ << ' '
            << config_.min_np << ' ' << config_.max_np << ' ';
        for (std::size_t k = 0; k < config_.weight_min.size(); ++k)
            out << config_.weight_min[k] << ' ' << config_.weight_max[k] << ' ';
        out << config_.terminal_scale << ' '
            << config_.train_mu_min << ' ' << config_.train_mu_max << '\n';
        actor_.save(out);
        if (!out) throw std::runtime_error("Failed writing DDPG policy");
    }
    void load(const std::filesystem::path& file) override {
        std::ifstream in(file, std::ios::binary);
        std::string magic;
        int nominal = 0, minimum = 0, maximum = 0;
        if (!(in >> magic >> nominal >> minimum >> maximum) ||
            magic != "DDPG_MPC_V15_ABSOLUTE")
            throw std::runtime_error("Policy uses an incompatible action definition; retrain with absolute Np/Q/R actions: " + file.string());
        std::array<double, 6> weight_min{}, weight_max{};
        for (std::size_t k = 0; k < weight_min.size(); ++k)
            if (!(in >> weight_min[k] >> weight_max[k]))
                throw std::runtime_error("Invalid policy Q/R bounds: " + file.string());
        double terminal_scale = 0.0, mumin = 0.0, mumax = 0.0;
        if (!(in >> terminal_scale >> mumin >> mumax) || nominal != nominal_np_ ||
            minimum != config_.min_np || maximum != config_.max_np ||
            weight_min != config_.weight_min || weight_max != config_.weight_max ||
            terminal_scale != config_.terminal_scale ||
            mumin != config_.train_mu_min || mumax != config_.train_mu_max)
            throw std::runtime_error("Policy RL Np/Q/R/mu bounds differ from this configuration: " + file.string());
        actor_.load(in);
        actor_target_ = actor_;
    }
};

struct EpisodeResult {
    double reward_sum = 0.0;
    double mean_reward = 0.0;
    double rmse = 0.0;
    double max_abs_error = 0.0;
    bool completed = false;
    int fallback_steps = 0;
    int grip_limited_steps = 0;
    std::set<int> np_values;
    int policy_decisions = 0;
    int changed_decisions = 0;
    double max_abs_normalized_weight_deviation = 0.0;
};

EpisodeResult runEpisode(const AppConfig& config, double speed, RlPolicy* policy,
                         bool training,
                         double requested_grip = std::numeric_limits<double>::quiet_NaN(),
                         double baseline_mean_reward = 0.0) {
    VehicleParameters vehicle = config.vehicle;
    vehicle.vx = speed;
        const auto native = config.output / "training_solver";
        auto staged = stageCarSimSimfile(config.carsim.simfile, native);
        const auto pars = native / "Run_all.par";
        std::ifstream input(pars);
        std::ostringstream buffer;
        buffer << input.rdbuf();
        std::string body = buffer.str();
        if (!input && !input.eof()) throw std::runtime_error("Cannot read staged CarSim Parsfile");
        const auto read_value = [&](const std::string& keyword) {
            std::istringstream source(body);
            std::string line;
            double result = std::numeric_limits<double>::quiet_NaN();
            int matches = 0;
            while (std::getline(source, line)) {
                std::istringstream fields(line);
                std::string name;
                fields >> name;
                if (name == keyword) {
                    fields >> result;
                    ++matches;
                }
            }
            if (matches != 1 || !std::isfinite(result))
                throw std::runtime_error("CarSim training requires one numeric " + keyword);
            return result;
        };
        const double source_grip = read_value("MU_ROAD_CONSTANT");
        const double source_stop = read_value("TSTOP");
        if (!std::isfinite(requested_grip)) requested_grip = source_grip;
        if (!(requested_grip > 0.0))
            throw std::invalid_argument("Training road friction must be positive");
        const double private_stop = std::max(source_stop,
            config.distance_m / std::max(1.0, speed * 0.75) + 5.0);
        const auto replace_value = [&](const std::string& keyword, double value) {
            std::istringstream source(body);
            std::ostringstream changed;
            std::string line;
            int matches = 0;
            while (std::getline(source, line)) {
                std::istringstream fields(line);
                std::string name;
                fields >> name;
                if (name == keyword) {
                    changed << keyword << ' ' << std::setprecision(12) << value << '\n';
                    ++matches;
                } else changed << line << '\n';
            }
            if (matches != 1)
                throw std::runtime_error("CarSim training requires one " + keyword);
            body = changed.str();
        };
        replace_value("SPEED_TARGET_CONSTANT", 3.6 * speed);
        replace_value("MU_ROAD_CONSTANT", requested_grip);
        replace_value("TSTOP", private_stop);
        std::ofstream output(pars, std::ios::binary | std::ios::trunc);
        output << body;
        output.close();
        if (!output) throw std::runtime_error("Cannot update private CarSim speed target");
        const auto reference = DoubleLaneChange::fromCarSimParsfile(pars);
        auto run_config = config.carsim;
        run_config.simfile = staged;
        run_config.sync_browser_result = false;
        CarSimPlant plant(std::move(run_config));
        const auto model = readCarSimMpcModel(native);
        if (model.speed_target_kmh.empty() ||
            std::abs(std::stod(model.speed_target_kmh) - 3.6 * speed) > 1e-5)
            throw std::runtime_error("CarSim solver speed target disagrees with private training Parsfile");
        if (std::abs(model.vehicle.road_grip_scale - requested_grip) > 1e-6)
            throw std::runtime_error("CarSim solver road friction differs from requested training mu");
        vehicle = model.vehicle;
        vehicle.vx = plant.initialState().vx;
        TireTable tires(model.tire_csv);
    std::map<std::tuple<int, double, double, double, double, double, double>,
             std::unique_ptr<LateralMpc>> controllers;
    VehicleState state = plant.initialState();
    int np = policy ? config.rl.nominal_np : config.mpc.np;
    double previous_steer = 0.0;
    bool grip_mode_active = false;
    EpisodeResult result;
    double squared_error = 0.0;
    int samples = 0;
    const int max_steps = static_cast<int>(2.5 * config.distance_m /
        (speed * config.mpc.dt)) + 1000;
    RlObservation decision_state{};
    MpcTuningAction decision_action = neutralRlAction(config.rl);
    decision_action.np = np;
    MpcTuningAction previous_action = decision_action;
    double block_reward = 0.0;
    int block_steps = 0;
    bool failed = false;
    for (int step = 0; step < max_steps && state.x < config.distance_m; ++step) {
        const int previous_np = np;
        const double prior_steer = previous_steer;
        if (policy && step % config.rl.decision_steps == 0) {
            previous_action = decision_action;
            decision_state = makeRlObservation(state, reference, np,
                config.distance_m, 30.0, config.rl, decision_action,
                vehicle.road_grip_scale);
            decision_action = policy->act(decision_state, training);
            decision_action.np = applyRlAction(config.mpc, decision_action,
                                                config.rl).np;
            ++result.policy_decisions;
            const std::array<double, 6> weights{{decision_action.q_y,
                decision_action.q_psi, decision_action.q_vy,
                decision_action.q_r, decision_action.r_steer,
                decision_action.r_delta}};
            bool weight_changed = false;
            for (std::size_t k = 0; k < weights.size(); ++k) {
                const double center = std::sqrt(config.rl.weight_min[k] *
                                                config.rl.weight_max[k]);
                const double normalized = std::log(weights[k] / center) /
                    (0.5 * std::log(config.rl.weight_max[k] / config.rl.weight_min[k]));
                result.max_abs_normalized_weight_deviation = std::max(
                    result.max_abs_normalized_weight_deviation, std::abs(normalized));
                weight_changed |= std::abs(normalized) > 1e-6;
            }
            if (decision_action.np != config.rl.nominal_np || weight_changed)
                ++result.changed_decisions;
            np = decision_action.np;
            block_reward = 0.0;
            block_steps = 0;
        }
        result.np_values.insert(np);
        auto point = reference.at(state.x);
        const double error = state.y - point.y;
        squared_error += error * error;
        result.max_abs_error = std::max(result.max_abs_error, std::abs(error));
        ++samples;
        try {
            const auto key = std::make_tuple(np, decision_action.q_y,
                decision_action.q_psi, decision_action.q_vy,
                decision_action.q_r, decision_action.r_steer,
                decision_action.r_delta);
            auto& controller = controllers[key];
            if (!controller) {
                auto settings = config.mpc;
                if (policy) settings = applyRlAction(config.mpc, decision_action, config.rl);
                controller = std::make_unique<LateralMpc>(vehicle, settings, reference, tires);
            }
            controller->setGripLimitedMode(grip_mode_active);
            MpcResult command;
            try { command = controller->control(state, previous_steer); }
            catch (const std::exception&) {
                command = controller->enterGripLimitedMode(state, previous_steer);
            }
            grip_mode_active = controller->gripLimitedModeActive();
            if ((!command.converged && !command.grip_limited) ||
                !std::isfinite(command.steer))
                throw std::runtime_error("MPC solve failed");
            if (command.grip_limited) ++result.grip_limited_steps;
            plant.step(state, command.steer, config.mpc.dt);
            if (!std::isfinite(state.x) || !std::isfinite(state.y) ||
                !std::isfinite(state.vx) || std::abs(state.y) > 20.0)
                throw std::runtime_error("Vehicle left numerical/safety range");
            previous_steer = command.steer;
            const auto next_ref = reference.at(state.x);
            const double ey = state.y - next_ref.y;
            result.max_abs_error = std::max(result.max_abs_error, std::abs(ey));
            const double epsi = angleDifference(state.psi, next_ref.psi);
            const double reward = scalarizedRlReward(
                ey, epsi, command.steer, np, config.mpc, config.rl,
                prior_steer, previous_np, decision_action, previous_action,
                state.vy, state.r - state.vx * next_ref.curvature,
                command.grip_limited);
            result.reward_sum += reward;
            block_reward += reward;
            ++block_steps;
        } catch (const std::exception& e) {
            (void)e;
            result.reward_sum -= config.rl.reward.reward_scale *
                                 config.rl.reward.failure_penalty;
            block_reward -= config.rl.reward.reward_scale *
                            config.rl.reward.failure_penalty;
            ++block_steps;
            ++result.fallback_steps;
            failed = true;
        }
        previous_action = decision_action;
        const bool done = failed || state.x >= config.distance_m || step + 1 == max_steps;
        if (done && state.x < config.distance_m) {
            const double penalty = config.rl.reward.reward_scale *
                                   config.rl.reward.incomplete_episode_penalty;
            result.reward_sum -= penalty;
            block_reward -= penalty;
        }
        if (policy && (done || (step + 1) % config.rl.decision_steps == 0) && training) {
            const auto next = makeRlObservation(state, reference, np, config.distance_m,
                                                30.0, config.rl, decision_action,
                                                vehicle.road_grip_scale);
            // A paired RL-neutral MPC run at the same speed and grip supplies a
            // scenario baseline. This removes much of the reward offset caused
            // by difficulty while retaining each action's relative outcome.
            policy->observe(decision_state, decision_action,
                            clip((block_reward / std::max(1, block_steps) -
                                  baseline_mean_reward) /
                                 config.rl.reward.reward_scale, -20.0, 20.0),
                            next, done);
        }
        if (failed) break;
    }
    result.completed = state.x >= config.distance_m && !failed;
    result.rmse = samples ? std::sqrt(squared_error / samples) : 0.0;
    result.mean_reward = samples ? result.reward_sum / samples :
                         -config.rl.reward.reward_scale *
                         (config.rl.reward.failure_penalty +
                          config.rl.reward.incomplete_episode_penalty);
    return result;
}

void chart(const std::filesystem::path& file, const std::string& title,
           const std::string& xlabel, const std::string& ylabel,
           const std::vector<std::vector<std::pair<double, double>>>& traces,
           const std::vector<std::string>& labels,
           bool integer_y = false, bool integer_x = false,
           bool comparison_style = false) {
    double xmin = std::numeric_limits<double>::infinity(), xmax = -xmin;
    double ymin = xmin, ymax = -xmin;
    for (const auto& tr : traces) for (const auto& xy : tr) {
        xmin = std::min(xmin, xy.first); xmax = std::max(xmax, xy.first);
        ymin = std::min(ymin, xy.second); ymax = std::max(ymax, xy.second);
    }
    if (!std::isfinite(xmin)) throw std::runtime_error("Chart has no samples");
    if (xmax <= xmin) xmax = xmin + 1.0;
    if (ymax <= ymin) { ymin -= 1.0; ymax += 1.0; }
    if (integer_y) {
        ymin = std::floor(ymin) - 1.0;
        ymax = std::ceil(ymax) + 1.0;
    } else {
        const double pad = 0.07 * (ymax - ymin); ymin -= pad; ymax += pad;
    }
    constexpr double plot_left = 190.0;
    constexpr double plot_right = 1070.0;
    constexpr double plot_width = plot_right - plot_left;
    constexpr double tick_label_x = plot_left - 16.0;
    const auto px = [&](double x) {
        return plot_left + plot_width * (x - xmin) / (xmax - xmin);
    };
    const auto py = [&](double y) { return 520.0 - 400.0 * (y - ymin) / (ymax - ymin); };
    std::ofstream out(file, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create chart: " + file.string());
    out << std::fixed << std::setprecision(3)
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1160\" height=\"620\" viewBox=\"0 0 1160 620\">"
        << "<rect width=\"1160\" height=\"620\" fill=\"white\"/>"
        << "<text x=\"580\" y=\"46\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"25\">"
        << title << "</text><rect x=\"" << plot_left
        << "\" y=\"120\" width=\"" << plot_width
        << "\" height=\"400\" fill=\"#f8fafc\" stroke=\"#94a3b8\"/>";
    for (int i = 0; i <= 5; ++i) {
        const double x = integer_x
            ? std::round(xmin + (xmax - xmin) * i / 5.0)
            : xmin + (xmax - xmin) * i / 5.0;
        out << "<line x1=\"" << px(x) << "\" x2=\"" << px(x)
            << "\" y1=\"120\" y2=\"520\" stroke=\"#e2e8f0\"/>"
            << "<text x=\"" << px(x) << "\" y=\"545\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">";
        if (integer_x) out << static_cast<int>(x);
        else out << x;
        out << "</text>";
    }
    if (integer_y) {
        const int step = std::max(1, static_cast<int>(std::ceil((ymax - ymin) / 5.0)));
        const int first = static_cast<int>(std::ceil(ymin / step)) * step;
        for (int y = first; y <= ymax; y += step)
            out << "<line x1=\"" << plot_left << "\" x2=\"" << plot_right
                << "\" y1=\"" << py(y)
                << "\" y2=\"" << py(y) << "\" stroke=\"#e2e8f0\"/>"
                << "<text x=\"" << tick_label_x << "\" y=\"" << py(y) + 5
                << "\" text-anchor=\"end\" font-family=\"Arial\" font-size=\"14\">"
                << y << "</text>";
    } else {
        for (int i = 0; i <= 5; ++i) {
            const double y = ymin + (ymax - ymin) * i / 5.0;
            out << "<line x1=\"" << plot_left << "\" x2=\"" << plot_right
                << "\" y1=\"" << py(y)
                << "\" y2=\"" << py(y) << "\" stroke=\"#e2e8f0\"/>"
                << "<text x=\"" << tick_label_x << "\" y=\"" << py(y) + 5
                << "\" text-anchor=\"end\" font-family=\"Arial\" font-size=\"14\">"
                << y << "</text>";
        }
    }
    const char* colors[] = {"#2563eb", "#dc2626", "#16a34a", "#a855f7"};
    for (std::size_t k = 0; k < traces.size(); ++k) {
        const char* color = colors[k % 4];
        const char* dash = nullptr;
        if (comparison_style) {
            if (labels[k] == "Reference") {
                color = "#0f766e";
                dash = "2,6";
            } else if (labels[k] == "Fixed Np") {
                color = "#1d4ed8";
                dash = "15,8";
            } else if (labels[k] == "DDPG Np") {
                color = "#be123c";
                dash = "10,5,2,5";
            }
        }
        out << "<polyline fill=\"none\" stroke=\"" << color
            << "\" stroke-width=\"2.8\" stroke-linecap=\"round\"";
        if (dash) out << " stroke-dasharray=\"" << dash << "\"";
        out << " points=\"";
        for (const auto& xy : traces[k]) out << px(xy.first) << ',' << py(xy.second) << ' ';
        out << "\"/><line x1=\"" << plot_left + 230 * k
            << "\" x2=\"" << plot_left + 30 + 230 * k
            << "\" y1=\"82\" y2=\"82\" stroke=\"" << color
            << "\" stroke-width=\"3\"";
        if (dash) out << " stroke-dasharray=\"" << dash << "\"";
        out << "/><text x=\"" << plot_left + 38 + 230 * k
            << "\" y=\"87\" font-family=\"Arial\" font-size=\"16\">" << labels[k] << "</text>";
    }
    out << "<text x=\"" << (plot_left + plot_right) / 2.0
        << "\" y=\"590\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"18\">"
        << xlabel << "</text><text transform=\"translate(36,320) rotate(-90)\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"18\">"
        << ylabel << "</text></svg>";
}
} // namespace

double scalarizedRlReward(double ey, double epsi, double steer, int np,
                          const MpcParameters& mpc, const RlConfig& config,
                          double previous_steer, int previous_np,
                          const MpcTuningAction& action,
                          const MpcTuningAction& previous_action,
                          double vy, double yaw_error, bool grip_limited) {
    const auto& r = config.reward;
    const double normalized_ey = clip(ey / r.ey_scale_m, -20.0, 20.0);
    const double normalized_epsi = clip(epsi / r.epsi_scale_rad, -20.0, 20.0);
    const double peak_excess = std::max(0.0, std::abs(ey) - r.peak_threshold_m) /
                               r.peak_threshold_m;
    // Quadratic near the path; linear tails preserve a learning signal when
    // the car is far from the path instead of saturating at a fixed penalty.
    const auto tracking_loss = [](double error) {
        const double magnitude = std::abs(error);
        return magnitude <= 1.0 ? error * error : 2.0 * magnitude - 1.0;
    };
    // A quadratic lateral tail through two normalized error units makes peak
    // deviations costly; beyond that, a linear tail avoids reward saturation.
    const auto lateral_loss = [](double error) {
        const double magnitude = std::abs(error);
        return magnitude <= 2.0 ? error * error : 4.0 * magnitude - 4.0;
    };
    const double normalized_steer = clip(steer / mpc.max_steer, -1.0, 1.0);
    const double normalized_steer_rate = clip((steer - previous_steer) /
        (mpc.max_steer_rate * mpc.dt), -2.0, 2.0);
    const double horizon_fraction = clip((np - config.min_np) /
        static_cast<double>(config.max_np - config.min_np), 0.0, 1.0);
    const double horizon_change = previous_np < config.min_np ? 0.0 :
        clip(std::abs(np - previous_np) /
             static_cast<double>(config.max_np - config.min_np), 0.0, 1.0);
    const std::array<double, 6> weights{action.q_y, action.q_psi,
        action.q_vy, action.q_r, action.r_steer, action.r_delta};
    const std::array<double, 6> old_weights{previous_action.q_y,
        previous_action.q_psi, previous_action.q_vy,
        previous_action.q_r, previous_action.r_steer,
        previous_action.r_delta};
    double deviation = 0.0, change = 0.0;
    for (std::size_t k = 0; k < weights.size(); ++k) {
        const double center = std::sqrt(config.weight_min[k] * config.weight_max[k]);
        const double log_span = 0.5 * std::log(config.weight_max[k] / config.weight_min[k]);
        const double log_weight = clip(std::log(weights[k] / center) / log_span, -1.0, 1.0);
        deviation += log_weight * log_weight / weights.size();
        const double log_change = clip(std::log(weights[k] / old_weights[k]) / log_span, -2.0, 2.0);
        change += log_change * log_change / weights.size();
    }
    const double loss = r.w_y * lateral_loss(normalized_ey)
        + r.w_peak * peak_excess * peak_excess
        + r.w_psi * tracking_loss(normalized_epsi)
        + r.w_steer * normalized_steer * normalized_steer
        + r.w_steer_rate * normalized_steer_rate * normalized_steer_rate
        + r.w_horizon * horizon_fraction
        + r.w_np_change * horizon_change * horizon_change
        + r.w_vy * std::pow(clip(vy / 2.0, -3.0, 3.0), 2)
        + r.w_yaw_error * std::pow(clip(yaw_error / 0.25, -3.0, 3.0), 2)
        + r.w_weight_change * change
        + r.w_weight_deviation * deviation
        + (grip_limited ? r.w_grip_limited : 0.0);
    return -r.reward_scale * std::min(loss, 120.0);
}

std::unique_ptr<RlPolicy> createRlPolicy(const RlConfig& config) {
    if (config.algorithm == "ddpg") return std::make_unique<DdpgPolicy>(config);
    throw std::invalid_argument("Unknown RL algorithm: " + config.algorithm);
}

MpcTuningAction neutralRlAction(const RlConfig& config) {
    MpcTuningAction action;
    action.np = config.nominal_np;
    action.q_y = std::sqrt(config.weight_min[0] * config.weight_max[0]);
    action.q_psi = std::sqrt(config.weight_min[1] * config.weight_max[1]);
    action.q_vy = std::sqrt(config.weight_min[2] * config.weight_max[2]);
    action.q_r = std::sqrt(config.weight_min[3] * config.weight_max[3]);
    action.r_steer = std::sqrt(config.weight_min[4] * config.weight_max[4]);
    action.r_delta = std::sqrt(config.weight_min[5] * config.weight_max[5]);
    return action;
}

RlObservation makeRlObservation(const VehicleState& state,
                                const DoubleLaneChange& reference,
                                int current_np, double distance_m,
                                double speed_scale, const RlConfig& config,
                                const MpcTuningAction& current_action,
                                double road_grip) {
    if (!(std::isfinite(road_grip) && road_grip > 0.0))
        throw std::invalid_argument("RL observation requires positive road grip");
    const auto point = reference.at(state.x);
    const double preview_m = std::min(std::max(0.0, distance_m - state.x),
        std::max(5.0, state.vx * config.curvature_preview_s));
    double peak_signed_curvature = point.curvature;
    double curvature_change = 0.0;
    for (int i = 1; i <= 12; ++i) {
        const double future = reference.at(state.x + preview_m * i / 12.0).curvature;
        if (std::abs(future) > std::abs(peak_signed_curvature))
            peak_signed_curvature = future;
        curvature_change = std::max(curvature_change,
                                     std::abs(future - point.curvature));
    }
    const auto scale_state = [](double value, double minimum, double maximum) {
        return clip(2.0 * (std::log(value) - std::log(minimum)) /
            (std::log(maximum) - std::log(minimum)) - 1.0, -1.0, 1.0);
    };
    return {clip(state.vx / speed_scale, 0.0, 2.0),
            clip(std::asinh((state.y - point.y) / config.reward.ey_scale_m), -4.0, 4.0),
            clip(std::asinh(angleDifference(state.psi, point.psi) /
                            config.reward.epsi_scale_rad), -4.0, 4.0),
            clip(state.vy / 5.0, -5.0, 5.0),
            clip(state.r, -5.0, 5.0),
            clip(point.curvature * 30.0, -5.0, 5.0),
            clip(peak_signed_curvature * 30.0, -5.0, 5.0),
            clip(curvature_change * 30.0, 0.0, 5.0),
            clip(2.0 * (current_np - config.min_np) /
                 static_cast<double>(config.max_np - config.min_np) - 1.0, -1.0, 1.0),
            clip(state.vx * state.vx * std::abs(peak_signed_curvature) /
                 (9.81 * road_grip),
                 0.0, 3.0),
            scale_state(current_action.q_y, config.weight_min[0], config.weight_max[0]),
            scale_state(current_action.q_psi, config.weight_min[1], config.weight_max[1]),
            scale_state(current_action.q_vy, config.weight_min[2], config.weight_max[2]),
            scale_state(current_action.q_r, config.weight_min[3], config.weight_max[3]),
            scale_state(current_action.r_steer, config.weight_min[4], config.weight_max[4]),
            scale_state(current_action.r_delta, config.weight_min[5], config.weight_max[5]),
            clip(2.0 * (road_grip - config.train_mu_min) /
                 (config.train_mu_max - config.train_mu_min) - 1.0, -1.0, 1.0)};
}

MpcParameters applyRlAction(const MpcParameters& base, const MpcTuningAction& action,
                            const RlConfig& config) {
    const std::array<double, 6> weights{{action.q_y, action.q_psi,
        action.q_vy, action.q_r, action.r_steer, action.r_delta}};
    for (std::size_t k = 0; k < weights.size(); ++k)
        if (!std::isfinite(weights[k]) || weights[k] < config.weight_min[k] - 1e-9 ||
            weights[k] > config.weight_max[k] + 1e-9)
            throw std::invalid_argument("RL absolute Q/R weight is outside configured bounds");
    MpcParameters result = base;
    result.np = std::clamp(action.np, std::max(config.min_np, base.nc), config.max_np);
    result.q_y = action.q_y;
    result.q_psi = action.q_psi;
    result.q_vy = action.q_vy;
    result.q_r = action.q_r;
    result.r_steer = action.r_steer;
    result.r_delta = action.r_delta;
    result.terminal_scale = config.terminal_scale;
    return result;
}

void runRlTraining(const AppConfig& input_config) {
    AppConfig config = input_config;
    // Training's comparison and reward reference is an RL-owned neutral MPC,
    // so editing [mpc].np or [weights] does not alter the learned task.
    config.mpc = applyRlAction(config.mpc, neutralRlAction(config.rl), config.rl);
    std::filesystem::create_directories(config.output);
    std::cout << "Configuration: " << config.source_config_file.string() << '\n'
              << "RL reference Np=" << config.mpc.np
              << "; DDPG action range=" << config.rl.min_np << ".."
              << config.rl.max_np << '\n'
              << "Output: " << std::filesystem::absolute(config.output).string() << '\n';
    {
        std::ofstream manifest(config.output / "run_manifest.txt", std::ios::binary);
        manifest << "Source config: " << config.source_config_file.string() << '\n'
                 << "RL reference Np: " << config.mpc.np << '\n'
                 << "DDPG Np bounds: " << config.rl.min_np << ".."
                 << config.rl.max_np << '\n'
                 << "Episodes: " << config.rl.episodes << '\n'
                 << "Output: " << std::filesystem::absolute(config.output).string() << '\n';
    }
    const double train_min = config.rl.train_speed_min;
    const double train_max = config.rl.train_speed_max;
    stageCarSimSimfile(config.carsim.simfile, config.output / "training_solver");
    const auto info = readCarSimRunInfo(config.output / "training_solver" / "Run_all.par");
    if (info.speed_target_constant.empty())
        throw std::runtime_error("CarSim training requires SPEED_TARGET_CONSTANT");
    const double nominal_speed = std::stod(info.speed_target_constant) / 3.6;
    std::cout << "CarSim training: source target=" << 3.6 * nominal_speed
              << " km/h; sampled=" << 3.6 * train_min << ".."
              << 3.6 * train_max << " km/h, mu="
              << config.rl.train_mu_min << ".." << config.rl.train_mu_max << '\n';
    auto policy = createRlPolicy(config.rl);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(config.rl.seed));
    std::uniform_real_distribution<double> unit_distribution(0.0, 1.0);
    // Allocate more experience near the measured-speed model transition.
    constexpr std::array<double, 7> speed_bin_fraction{{
        0.0, 6.0/90.0, 15.0/90.0, 30.0/90.0, 50.0/90.0, 70.0/90.0, 1.0}};
    std::vector<std::pair<int, int>> scenario_bins;
    for (int repeat = 0; repeat < (config.rl.episodes + 29) / 30; ++repeat)
        for (int speed_bin = 0; speed_bin < 6; ++speed_bin)
            for (int grip_bin = 0; grip_bin < 5; ++grip_bin)
                scenario_bins.emplace_back(speed_bin, grip_bin);
    std::shuffle(scenario_bins.begin(), scenario_bins.end(), rng);
    const auto fixed_validation = runEpisode(config, nominal_speed, nullptr, false);
    std::string source_snapshot;
    double nominal_grip = 0.0;
    {
        const auto native = config.output / "training_solver";
        std::ifstream source(native / "Run_visualizer.par", std::ios::binary);
        source_snapshot.assign(std::istreambuf_iterator<char>(source),
                               std::istreambuf_iterator<char>());
        if (source_snapshot.empty())
            throw std::runtime_error("Cannot snapshot CarSim training source Parsfile");
        const auto model = readCarSimMpcModel(native);
        nominal_grip = model.vehicle.road_grip_scale;
        AppConfig snapshot = config;
        snapshot.vehicle.mass = model.vehicle.mass;
        snapshot.vehicle.iz = model.vehicle.iz;
        snapshot.vehicle.a = model.vehicle.a;
        snapshot.vehicle.b = model.vehicle.b;
        snapshot.vehicle.road_grip_scale = model.vehicle.road_grip_scale;
        snapshot.vehicle.vx = nominal_speed;
        snapshot.tire_data_file = "tire_data.csv";
        std::ofstream effective(config.output / "effective_config.ini", std::ios::binary);
        effective << configToIni(snapshot);
        effective.close();
        if (!effective) throw std::runtime_error("Cannot save CarSim-derived training configuration");
        std::filesystem::copy_file(model.tire_csv, config.output / "tire_data.csv",
            std::filesystem::copy_options::overwrite_existing);
        std::filesystem::copy_file(native / "Run_visualizer.par",
            config.output / "training_source_Run_all.par",
            std::filesystem::copy_options::overwrite_existing);
    }
    const auto verify_source = [&] {
        std::ifstream source(config.output / "training_solver" / "Run_visualizer.par",
                             std::ios::binary);
        const std::string current{std::istreambuf_iterator<char>(source),
                                  std::istreambuf_iterator<char>()};
        if (current != source_snapshot)
            throw std::runtime_error("CarSim source Parsfile changed during DDPG training");
    };
    const double middle_grip = 0.5 * (config.rl.train_mu_min + config.rl.train_mu_max);
    const std::array<std::pair<double, double>, 13> validation_scenarios{{
        {train_min, config.rl.train_mu_min},
        {train_min, nominal_grip},
        {35.0 / 3.6, middle_grip},
        {36.0 / 3.6, nominal_grip},
        {40.0 / 3.6, nominal_grip},
        {45.0 / 3.6, nominal_grip},
        {50.0 / 3.6, nominal_grip},
        {60.0 / 3.6, nominal_grip},
        {70.0 / 3.6, nominal_grip},
        {75.0 / 3.6, middle_grip},
        {90.0 / 3.6, nominal_grip},
        {train_max, config.rl.train_mu_min},
        {train_max, config.rl.train_mu_max}}};
    std::vector<EpisodeResult> fixed_scenarios{fixed_validation};
    for (const auto [speed, grip] : validation_scenarios) {
        fixed_scenarios.push_back(runEpisode(config, speed, nullptr, false, grip));
        verify_source();
    }
    double fixed_validation_score = 0.0;
    int fixed_completed_scenarios = 0;
    for (const auto& run : fixed_scenarios) {
        fixed_validation_score += run.mean_reward;
        fixed_completed_scenarios += run.completed ? 1 : 0;
    }
    fixed_validation_score /= fixed_scenarios.size();
    std::vector<std::pair<double, double>> rewards, mean_rewards;
    std::vector<std::pair<double, double>> paired_advantages, paired_means;
    std::vector<std::pair<double, double>> checkpoint_scores, checkpoint_best,
                                            checkpoint_zero;
    std::vector<std::pair<double, double>> validation_rewards, validation_means;
    std::vector<std::pair<double, double>> fixed_reference;
    std::vector<std::pair<double, double>> advantages, advantage_means, zero_reference;
    double best_selection_score = fixed_validation_score;
    int best_completed_scenarios = fixed_completed_scenarios;
    int best_episode = 0;
    struct ValidationCase {
        double speed_kmh;
        double mu;
        bool completed;
        double rmse_m;
        double max_abs_error_m;
        double mean_reward;
        int grip_limited_steps;
    };
    std::vector<ValidationCase> best_cases;
    best_cases.push_back({3.6 * nominal_speed, nominal_grip,
        fixed_validation.completed, fixed_validation.rmse,
        fixed_validation.max_abs_error, fixed_validation.mean_reward,
        fixed_validation.grip_limited_steps});
    for (std::size_t i = 0; i < validation_scenarios.size(); ++i)
        best_cases.push_back({3.6 * validation_scenarios[i].first,
            validation_scenarios[i].second, fixed_scenarios[i + 1].completed,
            fixed_scenarios[i + 1].rmse, fixed_scenarios[i + 1].max_abs_error,
            fixed_scenarios[i + 1].mean_reward,
            fixed_scenarios[i + 1].grip_limited_steps});
    const auto best_model_file = config.output / "best_validation_policy.txt";
    policy->save(best_model_file); // deterministic actor initially equals fixed MPC
    std::ofstream log(config.output / "training_rewards.csv", std::ios::binary);
    if (!log) throw std::runtime_error("Cannot write training_rewards.csv");
    log << "episode,speed_kmh,mu,episode_return,episode_return_10,"
           "validation_return,validation_return_10,rl_reference_return,"
           "validation_advantage,validation_advantage_10,"
           "mean_step_reward,rmse_y_m,max_abs_y_m,completed,"
           "paired_rl_reference_mean_step_reward,paired_advantage_per_step,"
           "paired_advantage_10,exploration_sigma\n";
    std::ofstream checkpoint_log(config.output / "checkpoint_summary.csv", std::ios::binary);
    if (!checkpoint_log) throw std::runtime_error("Cannot write checkpoint_summary.csv");
    checkpoint_log << "episode,mean_reward_advantage_per_step,accepted,"
                      "completed_scenarios,acceptable_errors\n";
    std::ofstream scenario_log(config.output / "checkpoint_validation.csv", std::ios::binary);
    if (!scenario_log) throw std::runtime_error("Cannot write checkpoint_validation.csv");
    scenario_log << "episode,speed_kmh,mu,completed,rmse_y_m,max_abs_y_m,mean_step_reward,"
                    "episode_return,grip_limited_steps,fallback_steps\n";
    for (std::size_t i = 0; i < fixed_scenarios.size(); ++i) {
        const double case_speed = i == 0 ? nominal_speed : validation_scenarios[i - 1].first;
        const double case_grip = i == 0 ? nominal_grip : validation_scenarios[i - 1].second;
        const auto& fixed = fixed_scenarios[i];
        scenario_log << 0 << ',' << 3.6 * case_speed << ',' << case_grip << ','
                     << fixed.completed << ',' << fixed.rmse << ','
                     << fixed.max_abs_error << ',' << fixed.mean_reward
                     << ',' << fixed.reward_sum << ',' << fixed.grip_limited_steps
                     << ',' << fixed.fallback_steps << '\n';
    }
    for (int episode = 1; episode <= config.rl.episodes; ++episode) {
        const auto [speed_bin, grip_bin] = scenario_bins[episode - 1];
        const double speed_fraction = speed_bin_fraction[speed_bin] +
            unit_distribution(rng) *
            (speed_bin_fraction[speed_bin + 1] - speed_bin_fraction[speed_bin]);
        const double speed = train_min + speed_fraction * (train_max - train_min);
        const double grip = config.rl.train_mu_min +
            (grip_bin + unit_distribution(rng)) / 5.0 *
            (config.rl.train_mu_max - config.rl.train_mu_min);
        policy->setTrainingEpisode(episode, config.rl.episodes);
        const auto paired_fixed = runEpisode(config, speed, nullptr, false, grip);
        verify_source();
        const auto run = runEpisode(config, speed, policy.get(), true, grip,
                                    paired_fixed.mean_reward);
        verify_source();
        rewards.emplace_back(episode, run.reward_sum);
        paired_advantages.emplace_back(episode,
                                      run.mean_reward - paired_fixed.mean_reward);
        const int first = std::max(0, episode - 10);
        double mean = 0.0;
        for (int k = first; k < episode; ++k) mean += rewards[k].second;
        mean /= episode - first;
        mean_rewards.emplace_back(episode, mean);
        double paired_mean = 0.0;
        for (int k = first; k < episode; ++k)
            paired_mean += paired_advantages[k].second;
        paired_mean /= episode - first;
        paired_means.emplace_back(episode, paired_mean);
        const auto validation = runEpisode(config, nominal_speed, policy.get(),
                                           false, nominal_grip);
        verify_source();
        validation_rewards.emplace_back(episode, validation.reward_sum);
        double validation_mean = 0.0;
        for (int k = first; k < episode; ++k)
            validation_mean += validation_rewards[k].second;
        validation_mean /= episode - first;
        validation_means.emplace_back(episode, validation_mean);
        fixed_reference.emplace_back(episode, fixed_validation.reward_sum);
        advantages.emplace_back(episode,
            validation.reward_sum - fixed_validation.reward_sum);
        advantage_means.emplace_back(episode,
            validation_mean - fixed_validation.reward_sum);
        zero_reference.emplace_back(episode, 0.0);
        if (episode % 10 == 0 || episode == config.rl.episodes) {
            double score = validation.mean_reward;
            int completed_scenarios = validation.completed ? 1 : 0;
            bool acceptable_errors = validation.rmse <=
                fixed_validation.rmse * 1.05 + 0.02 &&
                validation.max_abs_error <=
                fixed_validation.max_abs_error * 1.10 + 0.05;
            std::vector<ValidationCase> checked_cases{{
                3.6 * nominal_speed, nominal_grip, validation.completed,
                validation.rmse, validation.max_abs_error,
                validation.mean_reward, validation.grip_limited_steps}};
            scenario_log << episode << ',' << 3.6 * nominal_speed << ',' << nominal_grip
                         << ',' << validation.completed << ',' << validation.rmse << ','
                         << validation.max_abs_error << ','
                         << validation.mean_reward << ',' << validation.reward_sum << ','
                         << validation.grip_limited_steps << ','
                         << validation.fallback_steps << '\n';
                std::size_t scenario_index = 1;
                for (const auto [case_speed, case_grip] : validation_scenarios) {
                    const auto check = runEpisode(config, case_speed, policy.get(),
                                                  false, case_grip);
                    verify_source();
                    score += check.mean_reward;
                    completed_scenarios += check.completed ? 1 : 0;
                    const auto& baseline = fixed_scenarios[scenario_index++];
                    const bool low_speed = 3.6 * case_speed <= 45.0 + 1e-6;
                    acceptable_errors &= check.rmse <=
                        baseline.rmse * (low_speed ? 1.05 : 1.10) +
                        (low_speed ? 0.005 : 0.05);
                    acceptable_errors &= check.max_abs_error <=
                        baseline.max_abs_error * (low_speed ? 1.02 : 1.10) +
                        (low_speed ? 0.005 : 0.05);
                    checked_cases.push_back({3.6 * case_speed, case_grip,
                        check.completed, check.rmse, check.max_abs_error,
                        check.mean_reward, check.grip_limited_steps});
                    scenario_log << episode << ',' << 3.6 * case_speed << ','
                                 << case_grip << ',' << check.completed << ','
                                 << check.rmse << ',' << check.max_abs_error << ','
                                 << check.mean_reward << ','
                                 << check.reward_sum << ',' << check.grip_limited_steps
                                 << ',' << check.fallback_steps << '\n';
                }
                score /= fixed_scenarios.size();
            const bool accepted = validation.completed && acceptable_errors &&
                completed_scenarios >= fixed_completed_scenarios &&
                score > best_selection_score + 0.01;
            if (accepted) {
                best_selection_score = score;
                best_completed_scenarios = completed_scenarios;
                best_episode = episode;
                best_cases = std::move(checked_cases);
                policy->save(best_model_file);
            }
            const double score_delta = score - fixed_validation_score;
            checkpoint_scores.emplace_back(episode, score_delta);
            checkpoint_best.emplace_back(episode,
                                         best_selection_score - fixed_validation_score);
            checkpoint_zero.emplace_back(episode, 0.0);
            checkpoint_log << episode << ',' << score_delta << ',' << accepted << ','
                           << completed_scenarios << ',' << acceptable_errors << '\n';
        }
        log << episode << ',' << 3.6 * speed << ',' << grip << ',' << run.reward_sum << ','
            << mean << ','
            << validation_rewards.back().second << ',' << validation_mean << ','
            << fixed_reference.back().second << ','
            << advantages.back().second << ',' << advantage_means.back().second
            << ',' << run.mean_reward << ','
            << run.rmse << ',' << run.max_abs_error << ','
            << (run.completed ? 1 : 0) << ',' << paired_fixed.mean_reward << ','
            << paired_advantages.back().second << ',' << paired_mean << ','
            << policy->explorationStdDev() << '\n';
        std::cout << "DDPG episode " << episode << '/' << config.rl.episodes
                  << " speed=" << 3.6 * speed << " km/h mu=" << grip
                  << " mean_reward=" << run.mean_reward
                  << " RMSE=" << run.rmse << " completed=" << run.completed << '\n';
    }
    log.flush();
    policy->load(best_model_file);
    if (best_episode == 0)
        std::cerr << "WARNING: No learned checkpoint beat the RL-neutral reference "
                     "without excessive tracking error; preserving the neutral policy.\n";
    policy->save(config.output / config.rl.model_file);
    {
        std::ofstream selected_log(config.output / "selected_validation.csv", std::ios::binary);
        if (!selected_log) throw std::runtime_error("Cannot write selected_validation.csv");
        selected_log << "seed,checkpoint_episode,controller,speed_kmh,mu,completed,"
                        "rmse_y_m,max_abs_y_m,mean_step_reward,grip_limited_steps\n";
        for (std::size_t i = 0; i < fixed_scenarios.size(); ++i) {
            const double case_speed = i == 0 ? nominal_speed :
                validation_scenarios[i - 1].first;
            const double case_grip = i == 0 ? nominal_grip :
                validation_scenarios[i - 1].second;
            const auto& fixed = fixed_scenarios[i];
            const auto& learned = best_cases[i];
            selected_log << config.rl.seed << ',' << best_episode << ",fixed,"
                         << 3.6 * case_speed << ',' << case_grip << ','
                         << fixed.completed << ',' << fixed.rmse << ','
                         << fixed.max_abs_error << ',' << fixed.mean_reward << ','
                         << fixed.grip_limited_steps << '\n';
            selected_log << config.rl.seed << ',' << best_episode << ",ddpg,"
                         << learned.speed_kmh << ',' << learned.mu << ','
                         << learned.completed << ',' << learned.rmse_m << ','
                         << learned.max_abs_error_m << ',' << learned.mean_reward << ','
                         << learned.grip_limited_steps << '\n';
        }
    }
    chart(config.output / "training_rewards.svg", "DDPG training episode return: 10-episode mean",
          "Episode", "Episode return (reward points)",
          {mean_rewards}, {"10-episode mean"}, false, true);
    chart(config.output / "training_episodes.svg", "DDPG exploratory episode returns",
          "Episode", "Episode return (reward points)",
          {rewards, mean_rewards}, {"Episode", "10-episode mean"}, false, true);
    chart(config.output / "paired_advantage.svg",
          "Training reward advantage over RL-neutral MPC at matched speed and grip",
          "Episode", "Reward advantage per step",
          {paired_advantages, paired_means, zero_reference},
          {"Paired advantage", "10-episode mean", "RL reference"}, false, true);
    chart(config.output / "checkpoint_convergence.svg",
          "Deterministic multi-scenario validation reward advantage",
          "Episode", "Mean reward advantage per step",
          {checkpoint_scores, checkpoint_best, checkpoint_zero},
          {"Current policy", "Best admissible", "RL reference"}, false, true);
    chart(config.output / "validation_returns.svg", "DDPG validation return at fixed speed",
          "Episode", "Episode return (reward points)",
          {validation_rewards, validation_means, fixed_reference},
          {"DDPG", "10-episode mean", "RL reference"}, false, true);
    chart(config.output / "validation_advantage.svg", "DDPG return minus RL reference",
          "Episode", "Return difference (reward points)",
          {advantages, advantage_means, zero_reference},
          {"DDPG advantage", "10-episode mean", "Equal return"}, false, true);
    if (best_episode > 0)
        std::cout << "Best validation checkpoint: episode " << best_episode
                  << ", score=" << best_selection_score << '\n';
    else
        std::cout << "Best validation checkpoint: RL-neutral reference (episode 0)\n";

        const auto selected = runEpisode(config, nominal_speed, policy.get(),
                                         false, nominal_grip);
        std::ofstream report(config.output / "rl_report.html", std::ios::binary);
        report << "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
               << "<title>CarSim DDPG training</title><style>body{font:16px Arial;max-width:1050px;margin:30px auto;color:#172033}img{width:100%;border:1px solid #ddd;margin:10px 0}table{border-collapse:collapse;width:100%}th,td{border:1px solid #cbd5e1;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}</style>"
               << "<h1>CarSim DDPG: 自适应 Np、Q 和 R</h1>"
               << "<p>MPC 在实测车速 32–40 km/h 平滑融合低速运动学模型与论文六状态三自由度动力学模型 [vy,vx,psi,r,Y,X]；"
                  "低速模型用 CarSim 实测侧偏角和横摆角速度残差校正。纵向轮胎滑移与刚度未映射，"
                  "由相邻 CarSim 状态的纵向加速度表示当前纵向作用。DDPG 在全部速度范围调节 Np/Q/R。</p>"
               << "<p>训练环境：CarSim VS Solver；车辆、道路和参考轨迹来自指定 Simfile 的 Parsfile。"
               << "每回合在私有副本改变 SPEED_TARGET_CONSTANT、MU_ROAD_CONSTANT 和必要的 TSTOP；原工况不修改。训练回合："
               << config.rl.episodes << "，随机种子：" << config.rl.seed
               << "，速度范围：" << 3.6 * train_min << "–" << 3.6 * train_max
               << " km/h；路面附着系数 μ 范围：" << config.rl.train_mu_min
               << "–" << config.rl.train_mu_max
               << "；原工况验证：" << 3.6 * nominal_speed << " km/h、μ="
               << nominal_grip << "。速度与 μ 采用 6×5 分层抽样，低速过渡区加密；当前位置与前方曲率从 CarSim 参考轨迹逐步计算。"
               << "曲率沿该轨迹变化，训练未随机生成其他道路轨迹。</p>"
               << "<p>动作：Np " << config.rl.min_np << "–" << config.rl.max_np
               << "，中性值 " << config.rl.nominal_np << "；Qy "
               << config.rl.weight_min[0] << "–" << config.rl.weight_max[0]
               << "；Qpsi " << config.rl.weight_min[1] << "–" << config.rl.weight_max[1]
               << "；Qvy " << config.rl.weight_min[2] << "–" << config.rl.weight_max[2]
               << "；Qr " << config.rl.weight_min[3] << "–" << config.rl.weight_max[3]
               << "；Rsteer " << config.rl.weight_min[4] << "–" << config.rl.weight_max[4]
               << "；Rdelta " << config.rl.weight_min[5] << "–" << config.rl.weight_max[5]
               << "。Q/R 是绝对数值，与固定 MPC 的 [weights] 无关。DDPG 在所有速度运行；低速动作仍可变化，"
                  "在 36 km/h 模型过渡中心附近平滑收紧，75 km/h 及以上为 1。"
                  "整数 Np 在小幅动作下可能不变。</p>"
               << "<p>奖励是横向及航向误差、侧向速度、横摆速率误差、转角、转角变化、"
               << "预测时域长度与跳变、Q/R 偏离及跳变、受限控制的加权负代价；"
               << "求解失败和未完成工况另加惩罚。每次训练还运行同车速、同附着的 RL 中性参考 MPC，"
                  "把每步回报差送入 Critic，以减小工况难度造成的回报偏移；"
                  "原始回报仍用于报告。探索噪声标准差逐回合从 0.25 降至 0.04。"
                  "策略通过经验回放和目标网络软更新训练，没有直接奖励权重变化。</p>"
               << "<p>每步奖励 = −" << config.rl.reward.reward_scale
               << " × min(120, 加权代价)。跟踪项：" << config.rl.reward.w_y
               << "·Ly(ey/" << config.rl.reward.ey_scale_m
               << ") + " << config.rl.reward.w_peak
               << "·[max(0,|ey|−" << config.rl.reward.peak_threshold_m
               << ")/" << config.rl.reward.peak_threshold_m << "]²"
               << " + " << config.rl.reward.w_psi << "·L(epsi/"
               << config.rl.reward.epsi_scale_rad
               << ")；Ly(z)=z²（|z|≤2），否则为 4|z|−4；"
                  "L(z)=z²（|z|≤1），否则为 2|z|−1，输入限于 ±20。侧向速度与横摆误差："
               << config.rl.reward.w_vy << "·clip(vy/2,±3)² + "
               << config.rl.reward.w_yaw_error << "·clip((r−vx·曲率)/0.25,±3)²。"
               << "转角、转角变化、Np 长度、Np 跳变系数依次为 "
               << config.rl.reward.w_steer << "、"
               << config.rl.reward.w_steer_rate << "、"
               << config.rl.reward.w_horizon << "、"
               << config.rl.reward.w_np_change << "。Q/R 相对独立搜索区间中心的对数变化与偏离 "
               << "系数分别为 " << config.rl.reward.w_weight_change << "、"
               << config.rl.reward.w_weight_deviation << "；受限控制系数 "
               << config.rl.reward.w_grip_limited << "。失败与未完成额外惩罚分别为 "
               << config.rl.reward.failure_penalty << " 和 "
               << config.rl.reward.incomplete_episode_penalty
               << " 个 reward_scale。详见生效配置与源码。</p>"
               << "<p>RL 中性参考横向 RMSE：" << fixed_validation.rmse
               << " m；最终策略横向 RMSE：" << selected.rmse
               << " m。基线/策略完成：" << fixed_validation.completed << "/"
               << selected.completed << "。最佳验证检查点："
               << (best_episode ? std::to_string(best_episode) : "0；保留 RL 中性策略")
               << "；该检查点完成 " << std::max(0, best_completed_scenarios)
               << "/" << fixed_scenarios.size() << " 个速度-附着验证工况。"
               << "完成整段仿真不代表轨迹误差达标；以下逐项列出横向 RMSE 和最大误差。"
               << "这些指标仅为本次车辆和轨迹验证，不代表其他工况的泛化性能。</p>"
               << "<h2>选中检查点的无探索验证</h2>"
               << "<table><tr><th>目标车速 (km/h)</th><th>路面 μ</th><th>完成</th>"
                  "<th>横向 RMSE (m)</th><th>最大横向误差 (m)</th><th>受限周期</th></tr>";
        for (const auto& check : best_cases)
            report << "<tr><td>" << check.speed_kmh << "</td><td>" << check.mu
                   << "</td><td>" << (check.completed ? "是" : "否")
                   << "</td><td>" << check.rmse_m << "</td><td>"
                   << check.max_abs_error_m << "</td><td>"
                   << check.grip_limited_steps << "</td></tr>";
        report << "</table><p>表中速度是 CarSim 的目标设定值；实际车速应查看对应仿真的"
                  " simulation.csv 或求解器输出。原工况以外的十三个工况用于检查点选择，"
                  "未向策略提供探索噪声。</p>"
               << "<p><a href=\"training_rewards.csv\">逐回合数据</a> · "
               << "<a href=\"effective_config.ini\">生效配置</a> · "
               << "<a href=\"training_source_Run_all.par\">工况原始快照</a> · "
               << "<a href=\"checkpoint_validation.csv\">速度-附着验证明细</a> · "
               << "<a href=\"checkpoint_summary.csv\">固定工况收敛记录</a> · "
               << "<a href=\"selected_validation.csv\">选中策略与 RL 中性参考逐工况对照</a> · "
               << "<a href=\"" << config.rl.model_file.generic_string()
               << "\">策略文件</a></p>"
               << "<p>随机训练回合工况不同，原始回报曲线不能直接判断收敛。"
                  "下图分别显示同工况配对优势和固定验证工况的无探索回报优势；"
                  "后者同时给出已通过误差约束的历史最佳值。历史最佳曲线单调不代表当前策略收敛。</p>"
               << "<img src=\"training_episodes.svg\"><img src=\"paired_advantage.svg\">"
               << "<img src=\"checkpoint_convergence.svg\"><img src=\"validation_returns.svg\">"
               << "<img src=\"validation_advantage.svg\"></html>";
        std::cout << "CarSim training report: "
                  << std::filesystem::absolute(config.output / "rl_report.html").string() << '\n';
}

void runRlHeldoutEvaluation(const AppConfig& config) {
    std::filesystem::create_directories(config.output);
    const auto policy_file = std::filesystem::absolute(config.rl.model_file);
    if (!std::filesystem::is_regular_file(policy_file))
        throw std::runtime_error("Held-out evaluation policy is missing: " + policy_file.string());
    const auto training_source = policy_file.parent_path() / "training_source_Run_all.par";
    if (!std::filesystem::is_regular_file(training_source))
        throw std::runtime_error("Held-out evaluation needs the policy's training source snapshot");
    auto policy = createRlPolicy(config.rl);
    policy->load(policy_file);
    std::filesystem::copy_file(policy_file, config.output / "policy_used.txt",
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(training_source, config.output / "training_source_Run_all.par",
                               std::filesystem::copy_options::overwrite_existing);
    const auto same_source = [&] {
        std::ifstream a(training_source, std::ios::binary);
        std::ifstream b(config.output / "training_solver" / "Run_visualizer.par",
                        std::ios::binary);
        return a && b && std::equal(std::istreambuf_iterator<char>(a),
                                    std::istreambuf_iterator<char>(),
                                    std::istreambuf_iterator<char>(b),
                                    std::istreambuf_iterator<char>());
    };
    constexpr std::array<std::pair<double, double>, 24> cases{{
        {30.0, 0.6}, {34.0, 0.6}, {35.0, 0.45},
        {36.0, 0.6}, {38.0, 0.6}, {40.0, 0.55}, {40.0, 0.65},
        {42.0, 0.6},
        {45.0, 0.55}, {45.0, 0.65},
        {47.0, 0.6},
        {50.0, 0.55}, {50.0, 0.65},
        {55.0, 0.6},
        {60.0, 0.55}, {60.0, 0.65},
        {65.0, 0.75},
        {70.0, 0.55}, {70.0, 0.65},
        {80.0, 0.55}, {80.0, 0.65},
        {90.0, 0.55}, {90.0, 0.65}, {120.0, 0.6}}};
    std::ofstream out(config.output / "heldout_validation.csv", std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write held-out validation CSV");
    out << "speed_kmh,mu,controller,completed,rmse_y_m,max_abs_y_m,"
           "mean_step_reward,grip_limited_steps,fallback_steps,"
           "np_min,np_max,np_unique,policy_decisions,changed_decisions,"
           "max_abs_normalized_weight_deviation\n";
    out << std::setprecision(12);
    for (const auto [speed_kmh, mu] : cases) {
        const auto fixed = runEpisode(config, speed_kmh / 3.6, nullptr, false, mu);
        if (!same_source())
            throw std::runtime_error("Held-out CarSim source differs from policy training source");
        const auto learned = runEpisode(config, speed_kmh / 3.6, policy.get(), false, mu);
        if (!same_source())
            throw std::runtime_error("Held-out CarSim source differs from policy training source");
        const auto write = [&](const char* name, const EpisodeResult& result) {
            out << speed_kmh << ',' << mu << ',' << name << ','
                << result.completed << ',' << result.rmse << ','
                << result.max_abs_error << ',' << result.mean_reward << ','
                << result.grip_limited_steps << ',' << result.fallback_steps << ','
                << *result.np_values.begin() << ',' << *result.np_values.rbegin() << ','
                << result.np_values.size() << ',' << result.policy_decisions << ','
                << result.changed_decisions << ','
                << result.max_abs_normalized_weight_deviation << '\n';
        };
        write("fixed", fixed);
        write("ddpg", learned);
        out.flush();
        std::cout << "Held-out " << speed_kmh << " km/h mu=" << mu
                  << " fixed RMSE/max=" << fixed.rmse << '/' << fixed.max_abs_error
                  << " DDPG RMSE/max=" << learned.rmse << '/'
                  << learned.max_abs_error << '\n';
    }
    std::ofstream config_file(config.output / "effective_config.ini", std::ios::binary);
    config_file << configToIni(config);
    std::cout << "Held-out results: "
              << std::filesystem::absolute(config.output / "heldout_validation.csv").string()
              << '\n';
}
