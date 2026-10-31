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
bool isPrioritySpeed(double speed_mps) {
    const double kmh = 3.6 * speed_mps;
    for (double priority : {60.0, 80.0, 95.0, 98.0, 100.0, 102.0, 104.0, 105.0, 116.0, 120.0})
        if (std::abs(kmh - priority) < 1e-6) return true;
    return false;
}

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
    struct Gradients {
        std::vector<std::vector<double>> w, b;
    };
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
    Gradients zeroGradients() const {
        Gradients result;
        for (const auto& layer : layers_) {
            result.w.emplace_back(layer.w.size(), 0.0);
            result.b.emplace_back(layer.b.size(), 0.0);
        }
        return result;
    }
    std::vector<double> backward(const std::vector<double>& input,
                                 std::vector<double> gradient,
                                 Gradients* accumulated = nullptr) const {
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
        for (int layer = static_cast<int>(layers_.size()) - 1; layer >= 0; --layer) {
            const auto& l = layers_[static_cast<std::size_t>(layer)];
            if (layer + 1 != static_cast<int>(layers_.size()) || bounded_output_)
                for (int j = 0; j < l.out; ++j)
                    gradient[j] *= 1.0 - activations[layer + 1][j] * activations[layer + 1][j];
            std::vector<double> previous(l.in, 0.0);
            for (int j = 0; j < l.out; ++j)
                for (int i = 0; i < l.in; ++i)
                    previous[i] += l.w[j * l.in + i] * gradient[j];
            if (accumulated) {
                for (int j = 0; j < l.out; ++j) {
                    for (int i = 0; i < l.in; ++i) {
                        const auto k = static_cast<std::size_t>(j * l.in + i);
                        accumulated->w[static_cast<std::size_t>(layer)][k] +=
                            gradient[j] * activations[layer][i];
                    }
                    accumulated->b[static_cast<std::size_t>(layer)][j] += gradient[j];
                }
            }
            gradient = std::move(previous);
        }
        return gradient;
    }
    std::vector<double> backward(const std::vector<double>& input,
                                 double output_gradient,
                                 Gradients* accumulated = nullptr) const {
        return backward(input, std::vector<double>{output_gradient}, accumulated);
    }
    void applyGradients(const Gradients& accumulated, double learning_rate,
                        double batch_size) {
        if (!(batch_size > 0.0)) throw std::invalid_argument("Empty network batch");
        double norm_squared = 0.0;
        for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
            for (const double value : accumulated.w[layer])
                norm_squared += value * value;
            for (const double value : accumulated.b[layer])
                norm_squared += value * value;
        }
        const double scale = std::min(1.0 / batch_size,
                                      5.0 / std::sqrt(norm_squared + 1e-12));
        ++updates_;
        const double correction1 = 1.0 - std::pow(0.9, static_cast<double>(updates_));
        const double correction2 = 1.0 - std::pow(0.999, static_cast<double>(updates_));
        auto adam = [&](double& parameter, double& m, double& v, double derivative) {
            derivative *= scale;
            m = 0.9 * m + 0.1 * derivative;
            v = 0.999 * v + 0.001 * derivative * derivative;
            parameter -= learning_rate * (m / correction1) /
                         (std::sqrt(v / correction2) + 1e-8);
        };
        for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
            auto& l = layers_[layer];
            for (std::size_t k = 0; k < l.w.size(); ++k)
                adam(l.w[k], l.mw[k], l.vw[k], accumulated.w[layer][k]);
            for (std::size_t k = 0; k < l.b.size(); ++k)
                adam(l.b[k], l.mb[k], l.vb[k], accumulated.b[layer][k]);
        }
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
    void saveTrainingState(std::ostream& out) const {
        out << updates_ << '\n';
        for (const auto& layer : layers_) {
            for (const auto* values : {&layer.w, &layer.b, &layer.mw, &layer.vw,
                                       &layer.mb, &layer.vb}) {
                for (double value : *values) out << value << ' ';
                out << '\n';
            }
        }
    }
    void loadTrainingState(std::istream& in) {
        if (!(in >> updates_)) throw std::runtime_error("Invalid network optimizer state");
        for (auto& layer : layers_)
            for (auto* values : {&layer.w, &layer.b, &layer.mw, &layer.vw,
                                 &layer.mb, &layer.vb})
                for (double& value : *values)
                    if (!(in >> value) || !std::isfinite(value))
                        throw std::runtime_error("Invalid network training state");
    }
    void rebaseInput(int index, double scale, double offset) {
        auto& first = layers_.front();
        if (index < 0 || index >= first.in)
            throw std::invalid_argument("Invalid actor observation index");
        for (int j = 0; j < first.out; ++j) {
            double& weight = first.w[static_cast<std::size_t>(j * first.in + index)];
            first.b[j] += weight * offset;
            weight *= scale;
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
    bool td3_;
    std::mt19937 rng_;
    Network actor_, actor_target_, critic_, critic_target_, critic2_, critic2_target_;
    std::vector<Transition> replay_;
    std::size_t replay_position_ = 0;
    std::size_t training_updates_ = 0;
    std::normal_distribution<double> noise_{0.0, 1.0};
    double exploration_sigma_ = 0.25;
    double exploration_start_ = 0.25;
    RlTrainingStats episode_stats_;
    std::string trainingIdentity() const {
        std::ostringstream out;
        out << std::setprecision(17) << config_.algorithm << ' ' << config_.action_profile
            << ' ' << nominal_np_ << ' ' << config_.min_np << ' ' << config_.max_np
            << ' ' << config_.decision_steps << ' ' << config_.terminal_scale
            << ' ' << config_.curvature_preview_s << ' '
            << config_.train_mu_min << ' ' << config_.train_mu_max;
        if (config_.relative_advantage) out << " relative_advantage_v1";
        if (config_.align_integer_np) out << " align_integer_np_v1";
        if (config_.direct_block_reward) out << " direct_block_reward_v1";
        if (config_.augmented_observation) out << " augmented_observation_v1";
        if (config_.actor_learning_rate != 0.0001) out << " actor_learning_rate_v1 " << config_.actor_learning_rate;
        if (config_.grip_demand_clip != 3.0) out << " grip_demand_clip_v1 " << config_.grip_demand_clip;
        if (config_.discount_gamma != 0.98) out << " discount_gamma_v1 " << config_.discount_gamma;
        if (config_.np_exploration_scale != 1.0) out << " np_exploration_scale_v1 " << config_.np_exploration_scale;
        if (config_.penalize_qp_failure) out << " qp_failure_penalty_v1";
        if (config_.strict_tracking_selection) out << " strict_tracking_selection_v1";
        if (config_.low_speed_blend) out << " low_speed_blend_v1 " << config_.blend_start_kmh << ' ' << config_.blend_end_kmh;
        if (config_.high_speed_blend) out << " high_speed_blend_v1 " << config_.high_blend_start_kmh << ' ' << config_.high_blend_end_kmh;
        for (std::size_t k = 0; k < config_.weight_min.size(); ++k)
            out << ' ' << config_.weight_min[k] << ' ' << config_.weight_max[k];
        const auto& r = config_.reward;
        for (double v : {r.reward_scale, r.ey_scale_m, r.epsi_scale_rad, r.w_y,
             r.w_peak, r.peak_threshold_m, r.w_psi, r.w_steer, r.w_steer_rate,
             r.w_horizon, r.w_np_change, r.w_vy, r.w_yaw_error,
             r.w_weight_change, r.w_weight_deviation, r.w_grip_limited,
             r.failure_penalty, r.incomplete_episode_penalty})
            out << ' ' << v;
        return out.str();
    }
    bool actionActive(int index) const {
        if (config_.action_profile == "np_only") return index == 0;
        return config_.action_profile == "full" ||
            index == 0 || index == 1 || index == 2 || index == 6;
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
        ContinuousAction result{encodeNp(action.np),
            encodeWeight(action.q_y, config_.weight_min[0], config_.weight_max[0]),
            encodeWeight(action.q_psi, config_.weight_min[1], config_.weight_max[1]),
            encodeWeight(action.q_vy, config_.weight_min[2], config_.weight_max[2]),
            encodeWeight(action.q_r, config_.weight_min[3], config_.weight_max[3]),
            encodeWeight(action.r_steer, config_.weight_min[4], config_.weight_max[4]),
            encodeWeight(action.r_delta, config_.weight_min[5], config_.weight_max[5])};
        for (int k = 0; k < 7; ++k)
            if (!actionActive(k)) result[k] = 0.0;
        return result;
    }
    MpcTuningAction decode(const ContinuousAction& values) const {
        const auto effective = [&](int index) {
            return actionActive(index) ? values[index] : 0.0;
        };
        MpcTuningAction action;
        action.np = decodeNp(effective(0));
        action.q_y = decodeWeight(effective(1), config_.weight_min[0], config_.weight_max[0]);
        action.q_psi = decodeWeight(effective(2), config_.weight_min[1], config_.weight_max[1]);
        action.q_vy = decodeWeight(effective(3), config_.weight_min[2], config_.weight_max[2]);
        action.q_r = decodeWeight(effective(4), config_.weight_min[3], config_.weight_max[3]);
        action.r_steer = decodeWeight(effective(5), config_.weight_min[4], config_.weight_max[4]);
        action.r_delta = decodeWeight(effective(6), config_.weight_min[5], config_.weight_max[5]);
        return action;
    }
    void update() {
        if (replay_.size() < 64) return;
        std::uniform_int_distribution<std::size_t> sample(0, replay_.size() - 1);
        // One mean minibatch update per replay insertion. The former 32 Adam
        // steps on individual samples made the critic chase scenario noise.
        auto critic_gradients = critic_.zeroGradients();
        auto critic2_gradients = critic2_.zeroGradients();
        for (int i = 0; i < 64; ++i) {
            const auto& tr = replay_[sample(rng_)];
            const auto next_vector = actor_target_.forward(vectorOf(tr.next));
            const double next_blend = policyActionBlend(tr.next[0] * 108.0, config_);
            ContinuousAction next_action{};
            for (int k = 0; k < 7; ++k) {
                const double smoothing = td3_ ?
                    clip(0.1 * noise_(rng_), -0.2, 0.2) : 0.0;
                next_action[k] = actionActive(k) ?
                    next_blend * clip(next_vector[k] + smoothing, -1.0, 1.0) : 0.0;
            }
            if (config_.align_integer_np && actionActive(0))
                next_action[0] = quantizeNormalizedNp(next_action[0], nominal_np_, config_.min_np, config_.max_np);
            double next_value = critic_target_.value(criticInput(tr.next, next_action));
            if (td3_) next_value = std::min(next_value,
                critic2_target_.value(criticInput(tr.next, next_action)));
            const double target = tr.reward + (tr.done ? 0.0 :
                config_.discount_gamma * next_value);
            const auto input = criticInput(tr.state, tr.action);
            const double prediction = critic_.value(input);
            const auto huber = [](double e) { const double a = std::abs(e); return a <= 2.0 ? 0.5 * e * e : 2.0 * (a - 1.0); };
            episode_stats_.critic1_loss_sum += huber(prediction - target) / 64.0;
            // Derivative of the Huber loss with threshold 2 reward units.
            critic_.backward(input, clip(prediction - target, -2.0, 2.0),
                             &critic_gradients);
            if (td3_) {
                const double second_prediction = critic2_.value(input);
                episode_stats_.critic2_loss_sum += huber(second_prediction - target) / 64.0;
                critic2_.backward(input,
                    clip(second_prediction - target, -2.0, 2.0),
                    &critic2_gradients);
            }
        }
        critic_.applyGradients(critic_gradients, 0.0003, 64.0);
        ++episode_stats_.critic_updates;
        if (td3_) critic2_.applyGradients(critic2_gradients, 0.0003, 64.0);
        if (++training_updates_ % 2 != 0) return;
        // Delayed, batch-averaged deterministic policy gradient. A small
        // action penalty discourages exploiting uncertain critic extremes.
        auto actor_gradients = actor_.zeroGradients();
        for (int i = 0; i < 32; ++i) {
            const auto& tr = replay_[sample(rng_)];
            const auto state = vectorOf(tr.state);
            const auto output = actor_.forward(state);
            const double blend = policyActionBlend(tr.state[0] * 108.0, config_);
            ContinuousAction action{};
            for (int k = 0; k < 7; ++k)
                action[k] = actionActive(k) ? blend * output[k] : 0.0;
            // Straight-through update: evaluate the critic at the same integer
            // horizon encoded in replay while retaining a surrogate gradient.
            if (config_.align_integer_np && actionActive(0))
                action[0] = quantizeNormalizedNp(action[0], nominal_np_, config_.min_np, config_.max_np);
            const auto input_gradient = critic_.backward(
                criticInput(tr.state, action), 1.0);
            std::vector<double> actor_gradient(7);
            // In easy segments the MPC reference already tracks well. Keep
            // the actor close to its neutral action unless lateral demand
            // justifies adaptation; this regularizes the same action terms
            // penalized by the paper's reward (31).
            const double demand = clip(tr.state[9] / 0.6, 0.0, 1.0);
            const double action_penalty = config_.action_profile == "np_only" ? 0.005 :
                0.05 + 0.25 * (1.0 - demand);
            double penalty = 0.0;
            for (int k = 0; k < 7; ++k) if (actionActive(k)) penalty += action_penalty * action[k] * action[k];
            episode_stats_.actor_objective_sum += (-critic_.value(criticInput(tr.state, action)) + penalty) / 32.0;
            for (int k = 0; k < 7; ++k)
                actor_gradient[k] = actionActive(k) ?
                    blend * (-clip(input_gradient[17 + k], -5.0, 5.0)
                    + 2.0 * action_penalty * action[k]) : 0.0;
            actor_.backward(state, actor_gradient, &actor_gradients);
        }
        actor_.applyGradients(actor_gradients, config_.actor_learning_rate, 32.0);
        ++episode_stats_.actor_updates;
        actor_target_.softUpdate(actor_, 0.005);
        critic_target_.softUpdate(critic_, 0.005);
        if (td3_) critic2_target_.softUpdate(critic2_, 0.005);
    }
public:
    explicit DdpgPolicy(const RlConfig& config)
        : config_(config), nominal_np_(config.nominal_np),
        td3_(config.algorithm == "td3"),
        rng_(static_cast<std::mt19937::result_type>(config.seed)),
        actor_(17, 64, 7, true, rng_), actor_target_(actor_),
        critic_(24, 64, 1, false, rng_), critic_target_(critic_),
        critic2_(24, 64, 1, false, rng_), critic2_target_(critic2_) {
        if (nominal_np_ <= config_.min_np || nominal_np_ >= config_.max_np)
            throw std::invalid_argument("DDPG nominal Np must lie inside action bounds");
        if (config_.action_profile == "np_only")
            exploration_sigma_ = exploration_start_ = 0.6;
        replay_.reserve(20000);
    }
    MpcTuningAction act(const RlObservation& observation, bool explore) override {
        auto values = actor_.forward(vectorOf(observation));
        const double blend = policyActionBlend(observation[0] * 108.0, config_);
        ContinuousAction action{};
        for (int k = 0; k < 7; ++k)
            action[k] = blend * clip(values[k] +
                (explore ? exploration_sigma_ * (k == 0 ? config_.np_exploration_scale : 1.0) * noise_(rng_) : 0.0), -1.0, 1.0);
        return decode(action);
    }
    void setTrainingEpisode(int episode, int total) override {
        const double progress = total <= 1 ? 1.0 :
            clip(static_cast<double>(episode - 1) / (total - 1), 0.0, 1.0);
        exploration_sigma_ = 0.04 + (exploration_start_ - 0.04) *
            (1.0 - progress) * (1.0 - progress);
    }
    double explorationStdDev() const override { return exploration_sigma_; }
    RlTrainingStats takeTrainingStats() override { const auto result = episode_stats_; episode_stats_ = {}; return result; }
    void observe(const RlObservation& state, const MpcTuningAction& action,
                 double reward, const RlObservation& next, bool done) override {
        ++episode_stats_.transitions;
        if (std::abs(reward) >= 8.0 - 1e-12) ++episode_stats_.clipped_rewards;
        const Transition tr{state, next, encode(action), reward, done};
        if (replay_.size() < 20000) replay_.push_back(tr);
        else { replay_[replay_position_] = tr; replay_position_ = (replay_position_ + 1) % 20000; }
        update();
    }
    void save(const std::filesystem::path& file) const override {
        std::ofstream out(file, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write DDPG policy: " + file.string());
        out << std::setprecision(17)
            << (config_.grip_demand_clip != 3.0 ?
                (td3_ ? "TD3_MPC_V21_COMPACT_GRIP_INCREMENTAL_LTV" : "DDPG_MPC_V21_COMPACT_GRIP_INCREMENTAL_LTV") :
                config_.augmented_observation ?
                (td3_ ? "TD3_MPC_V20_COMPACT_AUGMENTED_INCREMENTAL_LTV" : "DDPG_MPC_V20_COMPACT_AUGMENTED_INCREMENTAL_LTV") :
                config_.action_profile == "np_only" ?
                (td3_ ? "TD3_MPC_V19_NP_ONLY_INCREMENTAL_LTV" : "DDPG_MPC_V19_NP_ONLY_INCREMENTAL_LTV") :
                config_.action_profile == "compact" ?
                (td3_ ? "TD3_MPC_V18_COMPACT_INCREMENTAL_LTV" :
                        "DDPG_MPC_V18_COMPACT_INCREMENTAL_LTV") :
                (td3_ ? "TD3_MPC_V17_INCREMENTAL_LTV" :
                        "DDPG_MPC_V16_INCREMENTAL_LTV")) << ' '
            << nominal_np_ << ' '
            << config_.min_np << ' ' << config_.max_np << ' ';
        for (std::size_t k = 0; k < config_.weight_min.size(); ++k)
            out << config_.weight_min[k] << ' ' << config_.weight_max[k] << ' ';
        out << config_.terminal_scale << ' '
            << config_.train_mu_min << ' ' << config_.train_mu_max << '\n';
        actor_.save(out);
        if (!out) throw std::runtime_error("Failed writing DDPG policy");
    }
    void loadPolicy(const std::filesystem::path& file, bool warm_start) {
        std::ifstream in(file, std::ios::binary);
        std::string magic;
        int nominal = 0, minimum = 0, maximum = 0;
        if (!(in >> magic >> nominal >> minimum >> maximum))
            throw std::runtime_error("Invalid policy header: " + file.string());
        const auto expected_magic = config_.grip_demand_clip != 3.0 ?
                (td3_ ? "TD3_MPC_V21_COMPACT_GRIP_INCREMENTAL_LTV" : "DDPG_MPC_V21_COMPACT_GRIP_INCREMENTAL_LTV") :
                config_.augmented_observation ?
                (td3_ ? "TD3_MPC_V20_COMPACT_AUGMENTED_INCREMENTAL_LTV" : "DDPG_MPC_V20_COMPACT_AUGMENTED_INCREMENTAL_LTV") :
                config_.action_profile == "np_only" ?
                (td3_ ? "TD3_MPC_V19_NP_ONLY_INCREMENTAL_LTV" : "DDPG_MPC_V19_NP_ONLY_INCREMENTAL_LTV") :
                config_.action_profile == "compact" ?
                (td3_ ? "TD3_MPC_V18_COMPACT_INCREMENTAL_LTV" :
                        "DDPG_MPC_V18_COMPACT_INCREMENTAL_LTV") :
                (td3_ ? "TD3_MPC_V17_INCREMENTAL_LTV" :
                        "DDPG_MPC_V16_INCREMENTAL_LTV");
        const auto np_only_magic = td3_ ? "TD3_MPC_V19_NP_ONLY_INCREMENTAL_LTV" :
                                          "DDPG_MPC_V19_NP_ONLY_INCREMENTAL_LTV";
        const auto old_compact_magic = td3_ ? "TD3_MPC_V18_COMPACT_INCREMENTAL_LTV" : "DDPG_MPC_V18_COMPACT_INCREMENTAL_LTV";
        const bool old_compact_warm_start = warm_start && (config_.augmented_observation || config_.grip_demand_clip != 3.0) && magic == old_compact_magic;
        if (magic != expected_magic && !old_compact_warm_start && !(warm_start && !config_.augmented_observation && config_.action_profile == "compact" && magic == np_only_magic))
            throw std::runtime_error("Policy predates the incremental LTV-MPC controller; retrain it: " + file.string());
        std::array<double, 6> weight_min{}, weight_max{};
        for (std::size_t k = 0; k < weight_min.size(); ++k)
            if (!(in >> weight_min[k] >> weight_max[k]))
                throw std::runtime_error("Invalid policy Q/R bounds: " + file.string());
        double terminal_scale = 0.0, mumin = 0.0, mumax = 0.0;
        if (!(in >> terminal_scale >> mumin >> mumax) || nominal != nominal_np_ ||
            minimum != config_.min_np || maximum != config_.max_np ||
            weight_min != config_.weight_min || weight_max != config_.weight_max ||
            terminal_scale != config_.terminal_scale ||
            (!warm_start && (mumin != config_.train_mu_min || mumax != config_.train_mu_max)))
            throw std::runtime_error("Policy RL Np/Q/R/mu bounds differ from this configuration: " + file.string());
        actor_.load(in);
        if (old_compact_warm_start && config_.augmented_observation) {
            // These inputs were constant zero in the old compact mode.
            for (int index : {12, 13, 14}) actor_.rebaseInput(index, 0.0, 0.0);
        }
        if (warm_start && (mumin != config_.train_mu_min || mumax != config_.train_mu_max)) {
            if (mumin == mumax) {
                // A fixed-mu actor saw feature 16 as exactly zero throughout
                // training; remove its untrained random dependency before widening.
                actor_.rebaseInput(16, 0.0, 0.0);
            } else {
                if (config_.train_mu_min < mumin || config_.train_mu_max > mumax)
                    throw std::runtime_error("Warm start cannot extrapolate a variable-mu normalization beyond its original bounds");
                const double scale = (config_.train_mu_max - config_.train_mu_min) / (mumax - mumin);
                const double offset = (config_.train_mu_min + config_.train_mu_max - mumin - mumax) / (mumax - mumin);
                actor_.rebaseInput(16, scale, offset);
            }
        }
        actor_target_ = actor_;
    }
    void load(const std::filesystem::path& file) override { loadPolicy(file, false); }
    void warmStart(const std::filesystem::path& file) override {
        loadPolicy(file, true);
        exploration_start_ = 0.12;
    }
    void saveTrainingState(const std::filesystem::path& file) const override {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot write training state: " + file.string());
        out << std::setprecision(17) << "RL_TRAINING_STATE_V1\n"
            << std::quoted(trainingIdentity()) << '\n';
        for (const auto* network : {&actor_, &actor_target_, &critic_, &critic_target_,
                                    &critic2_, &critic2_target_})
            network->saveTrainingState(out);
        out << rng_ << '\n' << noise_ << '\n'
            << training_updates_ << ' ' << replay_position_ << ' '
            << exploration_sigma_ << ' ' << replay_.size() << '\n';
        for (const auto& tr : replay_) {
            for (double value : tr.state) out << value << ' ';
            for (double value : tr.action) out << value << ' ';
            out << tr.reward << ' ';
            for (double value : tr.next) out << value << ' ';
            out << (tr.done ? 1 : 0) << '\n';
        }
        if (!out) throw std::runtime_error("Failed writing training state: " + file.string());
    }
    void loadTrainingState(const std::filesystem::path& file) override {
        std::ifstream in(file, std::ios::binary);
        std::string magic, identity;
        if (!(in >> magic >> std::quoted(identity)) || magic != "RL_TRAINING_STATE_V1" ||
            identity != trainingIdentity())
            throw std::runtime_error("Training state algorithm, observation, action or reward configuration differs: " + file.string());
        for (auto* network : {&actor_, &actor_target_, &critic_, &critic_target_,
                              &critic2_, &critic2_target_})
            network->loadTrainingState(in);
        std::size_t count = 0;
        if (!(in >> rng_ >> noise_ >> training_updates_ >> replay_position_ >>
              exploration_sigma_ >> count) || count > 20000 || replay_position_ >= 20000 ||
            !std::isfinite(exploration_sigma_))
            throw std::runtime_error("Invalid replay or random state: " + file.string());
        replay_.clear();
        replay_.reserve(20000);
        for (std::size_t i = 0; i < count; ++i) {
            Transition tr{};
            for (double& value : tr.state)
                if (!(in >> value) || !std::isfinite(value)) throw std::runtime_error("Invalid replay state");
            for (double& value : tr.action)
                if (!(in >> value) || !std::isfinite(value)) throw std::runtime_error("Invalid replay action");
            if (!(in >> tr.reward) || !std::isfinite(tr.reward)) throw std::runtime_error("Invalid replay reward");
            for (double& value : tr.next)
                if (!(in >> value) || !std::isfinite(value)) throw std::runtime_error("Invalid replay next state");
            int done = 0;
            if (!(in >> done) || (done != 0 && done != 1)) throw std::runtime_error("Invalid replay terminal flag");
            tr.done = done != 0;
            replay_.push_back(tr);
        }
        exploration_start_ = clip(exploration_sigma_, 0.04, 0.15);
    }
};

struct EpisodeResult {
    double reward_sum = 0.0;
    double mean_reward = 0.0;
    std::vector<double> block_mean_rewards;
    double rmse = 0.0;
    double max_abs_error = 0.0;
    double rmse_heading_deg = 0.0;
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
                         const std::vector<double>* baseline_blocks = nullptr,
                         double station_scale = 1.0, int diagnostic_episode = 0) {
    VehicleParameters vehicle = config.vehicle;
    vehicle.vx = speed;
        const auto native = config.output / "training_solver";
        auto staged = stageCarSimSimfile(config.carsim.simfile, native);
        const auto pars = native / "Run_all.par";
        extendCarSimStationStop(pars, config.distance_m);
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
        const bool variable_grip = body.find("\nMU_ROAD_TABLE ") != std::string::npos;
        double source_grip = 0.0;
        if (variable_grip) {
            std::istringstream source(body); std::string line; bool table = false;
            while (std::getline(source, line)) {
                std::istringstream fields(line); std::string key; fields >> key;
                if (key == "MU_ROAD_TABLE") {
                    std::string kind; fields >> kind;
                    if (kind != "STEP" || table) throw std::runtime_error("Training requires one MU_ROAD_TABLE STEP");
                    table = true; continue;
                }
                if (table) {
                    char comma; std::istringstream row(line); double station;
                    if (!(row >> station >> comma >> source_grip) || comma != ',' || source_grip <= 0)
                        throw std::runtime_error("Invalid first road grip table row");
                    break;
                }
            }
            if (!(source_grip > 0)) throw std::runtime_error("Missing road grip table");
        } else source_grip = read_value("MU_ROAD_CONSTANT");
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
        if (config.rl.preserve_source_road) {
            // Keep the CarSim source friction values and transition locations unchanged.
        } else if (variable_grip) {
            std::istringstream source(body); std::ostringstream changed;
            std::string line; bool table = false; int count = 0;
            while (std::getline(source, line)) {
                std::istringstream fields(line); std::string key; fields >> key;
                if (key == "MU_ROAD_TABLE") { table = true; ++count; changed << line << '\n'; continue; }
                if (table && key == "ENDTABLE") { table = false; changed << line << '\n'; continue; }
                if (table) {
                    double station, mu; char comma; std::istringstream row(line);
                    if (!(row >> station >> comma >> mu) || comma != ',')
                        throw std::runtime_error("Invalid road grip table row in training source");
                    changed << std::setprecision(12) << station * station_scale << ", " << mu * requested_grip / source_grip << '\n';
                } else changed << line << '\n';
            }
            if (table || count != 1) throw std::runtime_error("Training requires one complete road grip table");
            body = changed.str();
        } else replace_value("MU_ROAD_CONSTANT", requested_grip);
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
        if (model.road_grip.variable() && config.rl.train_mu_min == config.rl.train_mu_max && !config.rl.preserve_source_road)
            throw std::runtime_error("Variable road requires a variable-mu policy/training range; fixed-mu policies cannot represent the profile");
        if (model.speed_target_kmh.empty() ||
            std::abs(std::stod(model.speed_target_kmh) - 3.6 * speed) > 1e-5)
            throw std::runtime_error("CarSim solver speed target disagrees with private training Parsfile");
        if (!config.rl.preserve_source_road && std::abs(model.road_grip.at(plant.initialState().x) - requested_grip) > 1e-6)
            throw std::runtime_error("CarSim initial grip differs from requested training mu; check road station, gain and offset");
        vehicle = model.vehicle;
        vehicle.vx = plant.initialState().vx;
        TireTable tires(model.tire_csv);
    std::map<std::tuple<int, double, double, double, double, double, double>,
             std::unique_ptr<LateralMpc>> controllers;
    LateralMpc* previous_controller = nullptr;
    VehicleState state = plant.initialState();
    int np = policy ? config.rl.nominal_np : config.mpc.np;
    double previous_steer = 0.0;
    bool grip_mode_active = false;
    EpisodeResult result;
    double squared_error = 0.0;
    double squared_heading_error = 0.0;
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
    std::ofstream diagnostic_steps, diagnostic_decisions;
    if (training && diagnostic_episode > 0) {
        diagnostic_steps.open(config.output / "diagnostic_steps.csv", std::ios::app);
        diagnostic_decisions.open(config.output / "diagnostic_decisions.csv", std::ios::app);
        if (!diagnostic_steps || !diagnostic_decisions) throw std::runtime_error("Cannot open training diagnostics");
        diagnostic_steps << std::setprecision(12);
        diagnostic_decisions << std::setprecision(12);
    }
    for (int step = 0; step < max_steps && state.x < config.distance_m; ++step) {
        vehicle.road_grip_scale = model.road_grip.at(state.x);
        const int previous_np = np;
        const double prior_steer = previous_steer;
        if (step % config.rl.decision_steps == 0) {
            block_reward = 0.0;
            block_steps = 0;
        }
        if (policy && step % config.rl.decision_steps == 0) {
            previous_action = decision_action;
            const double prior_plan = previous_controller && previous_controller->hasLastSolution() ?
                previous_controller->lastControlMoves()[std::min<std::size_t>(1, previous_controller->lastControlMoves().size() - 1)] : previous_steer;
            decision_state = makeRlObservation(state, reference, np,
                config.distance_m, 30.0, config.rl, decision_action,
                vehicle.road_grip_scale, previous_steer, grip_mode_active, prior_plan, config.mpc.max_steer);
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
        }
        result.np_values.insert(np);
        auto point = reference.at(state.x);
        const double error = state.y - point.y;
        squared_error += error * error;
        const double heading_error = angleDifference(state.psi, point.psi);
        squared_heading_error += heading_error * heading_error;
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
            if (previous_controller && previous_controller != controller.get())
                controller->adoptPreviousSolution(*previous_controller);
            previous_controller = controller.get();
            controller->setRoadGrip(vehicle.road_grip_scale);
            controller->setGripLimitedMode(grip_mode_active);
            MpcResult command;
            try { command = controller->control(state, previous_steer); }
            catch (const std::exception& ex) {
                command = controller->enterGripLimitedMode(state, previous_steer);
                command.prediction_failure = ex.what();
            }
            grip_mode_active = controller->gripLimitedModeActive();
            if ((!command.converged && !command.grip_limited) ||
                !std::isfinite(command.steer))
                throw std::runtime_error("MPC solve failed");
            if (command.grip_limited) ++result.grip_limited_steps;
            const bool qp_failure = optimizationFailed(command);
            if (qp_failure) {
                ++result.fallback_steps;
                if (config.rl.penalize_qp_failure) {
                    const double penalty = config.rl.reward.reward_scale * config.rl.reward.failure_penalty;
                    result.reward_sum -= penalty;
                    block_reward -= penalty;
                }
            }
            plant.step(state, command.steer, config.mpc.dt);
            if (!std::isfinite(state.x) || !std::isfinite(state.y) ||
                !std::isfinite(state.vx) || std::abs(state.y) > 20.0)
                throw std::runtime_error("Vehicle left numerical/safety range");
            previous_steer = command.steer;
            const auto next_ref = reference.at(state.x);
            const double ey = state.y - next_ref.y;
            result.max_abs_error = std::max(result.max_abs_error, std::abs(ey));
            const double epsi = angleDifference(state.psi, next_ref.psi);
            std::array<double, 11> reward_components{};
            const double reward = scalarizedRlReward(
                ey, epsi, command.steer, np, config.mpc, config.rl,
                prior_steer, previous_np, decision_action, previous_action,
                state.vy, state.r - state.vx * next_ref.curvature,
                command.grip_limited, diagnostic_steps.is_open() ? &reward_components : nullptr);
            if (diagnostic_steps.is_open()) {
                const double sum = std::accumulate(reward_components.begin(), reward_components.end(), 0.0);
                diagnostic_steps << diagnostic_episode << ',' << step << ',' << state.x << ',' << ey << ',' << epsi << ','
                    << reward << ',' << (sum >= 120.0) << ',' << qp_failure << ',' << command.grip_limited;
                for (double term : reward_components) diagnostic_steps << ',' << term;
                diagnostic_steps << '\n';
            }
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
        if (done || (step + 1) % config.rl.decision_steps == 0) {
            const double block_mean = block_reward / std::max(1, block_steps);
            const std::size_t block_index = result.block_mean_rewards.size();
            result.block_mean_rewards.push_back(block_mean);
            if (policy && training) {
                if (!baseline_blocks || baseline_blocks->empty())
                    throw std::runtime_error("Training requires matched baseline blocks");
                const auto next = makeRlObservation(state, reference, np, config.distance_m,
                                                    30.0, config.rl, decision_action,
                                                    model.road_grip.at(state.x), previous_steer, grip_mode_active,
                                                    previous_controller && previous_controller->hasLastSolution() ?
                                                        previous_controller->lastControlMoves()[std::min<std::size_t>(1, previous_controller->lastControlMoves().size() - 1)] : previous_steer,
                                                    config.mpc.max_steer);
                // Compare the same time/track segment, not an episode-wide
                // average that varies strongly across the lane-change phases.
                const double baseline = (*baseline_blocks)[std::min(
                    block_index, baseline_blocks->size() - 1)];
                const double train_reward = trainingBlockReward(block_mean, baseline, config.rl);
                if (diagnostic_decisions.is_open()) {
                    diagnostic_decisions << diagnostic_episode << ',' << block_index << ',' << state.x << ',' << block_mean << ','
                        << baseline << ',' << train_reward << ',' << (std::abs(train_reward) >= 8.0 - 1e-12) << ',' << done;
                    for (double value : decision_state) diagnostic_decisions << ',' << value;
                    diagnostic_decisions << ',' << decision_action.np << ',' << decision_action.q_y << ','
                        << decision_action.q_psi << ',' << decision_action.r_delta << '\n';
                }
                policy->observe(decision_state, decision_action, train_reward, next, done);
            }
        }
        if (failed) break;
    }
    result.completed = state.x >= config.distance_m && !failed;
    result.rmse = samples ? std::sqrt(squared_error / samples) : 0.0;
    result.rmse_heading_deg = samples ? std::sqrt(squared_heading_error / samples) * 180.0 / std::acos(-1.0) : 0.0;
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
    if (!std::isfinite(xmin)) {
        std::ofstream empty(file, std::ios::binary);
        if (!empty) throw std::runtime_error("Cannot create empty chart: " + file.string());
        empty << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1160\" height=\"300\">"
                 "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>"
                 "<text x=\"580\" y=\"70\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"25\">"
              << title << "</text><text x=\"580\" y=\"160\" text-anchor=\"middle\" "
                 "font-family=\"Arial\" font-size=\"20\">No complete cycle yet</text></svg>";
        return;
    }
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
                          double vy, double yaw_error, bool grip_limited,
                          std::array<double, 11>* components) {
    const auto& r = config.reward;
    const double normalized_ey = clip(ey / r.ey_scale_m, -20.0, 20.0);
    const double normalized_epsi = clip(epsi / r.epsi_scale_rad, -20.0, 20.0);
    const double peak_excess = std::max(0.0, std::abs(ey) - r.peak_threshold_m) /
                               r.peak_threshold_m;
    // Smooth absolute tracking error: the paper's normalized |e| term with a
    // finite derivative at zero. L(1)=1; large errors retain a linear signal.
    const auto tracking_loss = [](double error) {
        return (std::sqrt(1.0 + error * error) - 1.0) /
               (std::sqrt(2.0) - 1.0);
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
    const std::array<double, 11> terms{{r.w_y * tracking_loss(normalized_ey),
        r.w_peak * peak_excess * peak_excess, r.w_psi * tracking_loss(normalized_epsi),
        r.w_steer * normalized_steer * normalized_steer,
        r.w_steer_rate * normalized_steer_rate * normalized_steer_rate,
        r.w_horizon * horizon_fraction + r.w_np_change * horizon_change * horizon_change,
        r.w_vy * std::pow(clip(vy / 2.0, -3.0, 3.0), 2),
        r.w_yaw_error * std::pow(clip(yaw_error / 0.25, -3.0, 3.0), 2),
        r.w_weight_change * change, r.w_weight_deviation * deviation,
        grip_limited ? r.w_grip_limited : 0.0}};
    if (components) *components = terms;
    const double loss = std::accumulate(terms.begin(), terms.end(), 0.0);
    return -r.reward_scale * std::min(loss, 120.0);
}

std::unique_ptr<RlPolicy> createRlPolicy(const RlConfig& config) {
    if (config.algorithm == "ddpg" || config.algorithm == "td3")
        return std::make_unique<DdpgPolicy>(config);
    throw std::invalid_argument("Unknown RL algorithm: " + config.algorithm);
}

double pairedRewardAdvantage(double reward, double baseline, const RlConfig& config) {
    const double scale = config.relative_advantage ?
        std::max(0.2 * config.reward.reward_scale, std::abs(baseline)) :
        config.reward.reward_scale;
    return clip((reward - baseline) / scale, -8.0, 8.0);
}

double trainingBlockReward(double reward, double baseline, const RlConfig& config) {
    return config.direct_block_reward ? clip(reward / config.reward.reward_scale, -8.0, 8.0) :
        pairedRewardAdvantage(reward, baseline, config);
}

double quantizeNormalizedNp(double action, int nominal, int minimum, int maximum) {
    if (!(minimum < nominal && nominal < maximum)) throw std::invalid_argument("Invalid Np bounds");
    const int np = std::clamp(static_cast<int>(std::lround(nominal + clip(action, -1.0, 1.0) *
        (action >= 0.0 ? maximum - nominal : nominal - minimum))), minimum, maximum);
    return np >= nominal ? static_cast<double>(np - nominal) / (maximum - nominal) :
                           static_cast<double>(np - nominal) / (nominal - minimum);
}

double lowSpeedActionBlend(double speed_kmh, const RlConfig& config) {
    if (!config.low_speed_blend) return 1.0;
    return clip((std::abs(speed_kmh) - config.blend_start_kmh) /
                (config.blend_end_kmh - config.blend_start_kmh), 0.0, 1.0);
}

double policyActionBlend(double speed_kmh, const RlConfig& config) {
    const double low = lowSpeedActionBlend(speed_kmh, config);
    if (!config.high_speed_blend) return low;
    return low * clip((config.high_blend_end_kmh - std::abs(speed_kmh)) /
                      (config.high_blend_end_kmh - config.high_blend_start_kmh), 0.0, 1.0);
}

bool trackingSafetyAcceptable(double heading_rmse_deg, int grip_steps,
                              double reference_heading_rmse_deg, int reference_grip_steps) {
    return std::isfinite(heading_rmse_deg) && std::isfinite(reference_heading_rmse_deg) &&
        heading_rmse_deg <= reference_heading_rmse_deg * 1.05 + 0.02 &&
        grip_steps <= reference_grip_steps;
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
                                double road_grip, double previous_steer,
                                bool grip_mode, double previous_plan, double steer_scale) {
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
    if (!(steer_scale > 0.0)) throw std::invalid_argument("RL steering observation scale must be positive");
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
                 0.0, config.grip_demand_clip),
            scale_state(current_action.q_y, config.weight_min[0], config.weight_max[0]),
            scale_state(current_action.q_psi, config.weight_min[1], config.weight_max[1]),
            config.augmented_observation ? clip(previous_steer / steer_scale, -1.0, 1.0) : scale_state(current_action.q_vy, config.weight_min[2], config.weight_max[2]),
            config.augmented_observation ? (grip_mode ? 1.0 : 0.0) : scale_state(current_action.q_r, config.weight_min[3], config.weight_max[3]),
            config.augmented_observation ? clip(previous_plan / steer_scale, -1.0, 1.0) : scale_state(current_action.r_steer, config.weight_min[4], config.weight_max[4]),
            scale_state(current_action.r_delta, config.weight_min[5], config.weight_max[5]),
            config.train_mu_min == config.train_mu_max ? 0.0 :
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
    const std::string algorithm_name = config.rl.algorithm == "td3" ? "TD3" : "DDPG";
    // Training's comparison and reward reference is an RL-owned neutral MPC,
    // so editing [mpc].np or [weights] does not alter the learned task.
    config.mpc = applyRlAction(config.mpc, neutralRlAction(config.rl), config.rl);
    std::filesystem::create_directories(config.output);
    std::cout << "Configuration: " << config.source_config_file.string() << '\n'
              << "RL reference Np=" << config.mpc.np
              << "; " << algorithm_name << " action range=" << config.rl.min_np << ".."
              << config.rl.max_np << '\n'
              << "Output: " << std::filesystem::absolute(config.output).string() << '\n';
    {
        std::ofstream manifest(config.output / "run_manifest.txt", std::ios::binary);
        manifest << "Source config: " << config.source_config_file.string() << '\n'
                 << "RL reference Np: " << config.mpc.np << '\n'
                 << algorithm_name << " Np bounds: " << config.rl.min_np << ".."
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
    if (std::abs(train_max - train_min) < 1e-9 &&
        std::abs(nominal_speed - train_min) > 1e-6)
        throw std::runtime_error("Fixed-speed training requires the source CarSim speed target to match the training speed");
    std::cout << "CarSim training: source target=" << 3.6 * nominal_speed
              << " km/h; sampled=" << 3.6 * train_min << ".."
              << 3.6 * train_max << " km/h, mu="
              << config.rl.train_mu_min << ".." << config.rl.train_mu_max
              << (config.rl.preserve_source_road ? " (observation bounds; source road unchanged)" : "") << '\n';
    auto policy = createRlPolicy(config.rl);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(config.rl.seed));
    std::uniform_real_distribution<double> unit_distribution(0.0, 1.0);
    // Allocate more experience at low speed, where spatial preview is shorter.
    const std::vector<double> speed_bin_fraction = config.rl.focus_target_speeds ?
        std::vector<double>{0.0, 20.0/90.0, 25.0/90.0, 35.0/90.0,
                            45.0/90.0, 55.0/90.0, 65.0/90.0, 75.0/90.0, 1.0} :
        std::vector<double>{0.0, 6.0/90.0, 15.0/90.0, 30.0/90.0,
                            50.0/90.0, 70.0/90.0, 1.0};
    const bool fixed_grip_training =
        config.rl.train_mu_min == config.rl.train_mu_max;
    const int grip_bins = (fixed_grip_training || config.rl.preserve_source_road) ? 1 : 5;
    const int speed_bins = static_cast<int>(speed_bin_fraction.size()) - 1;
    const int cycle_size = speed_bins * grip_bins;
    std::vector<std::pair<int, int>> scenario_bins;
    for (int repeat = 0; repeat < (config.rl.episodes + cycle_size - 1) / cycle_size; ++repeat) {
        std::vector<std::pair<int, int>> cycle;
        for (int speed_bin = 0; speed_bin < speed_bins; ++speed_bin)
            for (int grip_bin = 0; grip_bin < grip_bins; ++grip_bin)
                cycle.emplace_back(speed_bin, grip_bin);
        std::shuffle(cycle.begin(), cycle.end(), rng);
        if (config.rl.curriculum)
            std::stable_sort(cycle.begin(), cycle.end(), [](const auto& a, const auto& b) {
                return a.first + a.second < b.first + b.second;
            });
        scenario_bins.insert(scenario_bins.end(), cycle.begin(), cycle.end());
    }
    auto fixed_validation = runEpisode(config, nominal_speed, nullptr, false,
        fixed_grip_training ? config.rl.train_mu_min :
            std::numeric_limits<double>::quiet_NaN());
    std::string source_snapshot;
    double nominal_grip = 0.0;
    double sampled_mu_min = config.rl.train_mu_min;
    double sampled_mu_max = config.rl.train_mu_max;
    {
        const auto native = config.output / "training_solver";
        std::ifstream source(native / "Run_visualizer.par", std::ios::binary);
        source_snapshot.assign(std::istreambuf_iterator<char>(source),
                               std::istreambuf_iterator<char>());
        if (source_snapshot.empty())
            throw std::runtime_error("Cannot snapshot CarSim training source Parsfile");
        const auto model = readCarSimMpcModel(native);
        nominal_grip = model.vehicle.road_grip_scale;
        if (config.rl.preserve_source_road) {
            if (config.rl.vary_road_transition)
                throw std::runtime_error("Cannot vary road transitions while preserving the source road");
            auto check_mu = [&](double mu) {
                if (mu < config.rl.train_mu_min - 1e-9 || mu > config.rl.train_mu_max + 1e-9)
                    throw std::runtime_error("Source road grip exceeds the policy observation range");
            };
            check_mu(nominal_grip);
            for (const auto& point : model.road_grip.table)
                check_mu(model.road_grip.at(point.first * model.road_grip.station_scale + model.road_grip.station_start));
        }
        if (config.rl.vary_road_transition && !model.road_grip.variable())
            throw std::runtime_error("Road transition variation requires a MU_ROAD_TABLE STEP source");
        if (model.road_grip.variable() && !config.rl.preserve_source_road) {
            if (std::abs(model.road_grip.offset + model.road_grip.lateral) > 1e-12)
                throw std::runtime_error("Variable-road training requires zero combined road offset for proportional scaling");
            double min_ratio = 1.0, max_ratio = 1.0;
            for (const auto& point : model.road_grip.table) {
                const double effective = point.second * model.road_grip.gain;
                min_ratio = std::min(min_ratio, effective / nominal_grip);
                max_ratio = std::max(max_ratio, effective / nominal_grip);
            }
            sampled_mu_min = std::max(sampled_mu_min, config.rl.train_mu_min / min_ratio);
            sampled_mu_max = std::min(sampled_mu_max, config.rl.train_mu_max / max_ratio);
            if (sampled_mu_min > sampled_mu_max + 1e-9)
                throw std::runtime_error("Road grip profile cannot fit within configured training mu bounds");
            std::cout << "Variable road: sampled initial mu=" << sampled_mu_min << ".."
                      << sampled_mu_max << "; full road remains within policy mu bounds.\n";
        }
        if (config.rl.preserve_source_road) sampled_mu_min = sampled_mu_max = nominal_grip;
        AppConfig snapshot = config;
        snapshot.vehicle.mass = model.vehicle.mass;
        snapshot.vehicle.iz = model.vehicle.iz;
        snapshot.vehicle.a = model.vehicle.a;
        snapshot.vehicle.b = model.vehicle.b;
        snapshot.vehicle.road_grip_scale = fixed_grip_training ?
            config.rl.train_mu_min : model.vehicle.road_grip_scale;
        snapshot.vehicle.vx = nominal_speed;
        snapshot.tire_data_file = "tire_data.csv";
        std::ofstream effective(config.output / "effective_config.ini", std::ios::binary);
        effective << configToIni(snapshot);
        effective.close();
        if (!effective) throw std::runtime_error("Cannot save CarSim-derived training configuration");
        std::ofstream revision(config.output / "controller_revision.txt", std::ios::binary);
        revision << "ltv-warm-sequence-v1\n";
        if (!revision) throw std::runtime_error("Cannot save controller revision");
        std::filesystem::copy_file(model.tire_csv, config.output / "tire_data.csv",
            std::filesystem::copy_options::overwrite_existing);
        std::filesystem::copy_file(native / "Run_visualizer.par",
            config.output / "training_source_Run_all.par",
            std::filesystem::copy_options::overwrite_existing);
    }
    // Fixed-grip training must select checkpoints on the same grip that
    // generated its replay. The archived source still records its original mu.
    if (fixed_grip_training) nominal_grip = config.rl.train_mu_min;
    else if (!config.rl.preserve_source_road && (nominal_grip < config.rl.train_mu_min || nominal_grip > config.rl.train_mu_max)) {
        nominal_grip = clip(nominal_grip, config.rl.train_mu_min, config.rl.train_mu_max);
        fixed_validation = runEpisode(config, nominal_speed, nullptr, false, nominal_grip);
    }
    const auto verify_source = [&] {
        std::ifstream source(config.output / "training_solver" / "Run_visualizer.par",
                             std::ios::binary);
        const std::string current{std::istreambuf_iterator<char>(source),
                                  std::istreambuf_iterator<char>()};
        if (current != source_snapshot)
            throw std::runtime_error("CarSim source Parsfile changed during DDPG training");
    };
    const double middle_grip = 0.5 * (sampled_mu_min + sampled_mu_max);
    const auto fixed_validation_speed = [&](double original_kmh) {
        return train_min + (original_kmh - 30.0) / 90.0 *
            (train_max - train_min);
    };
    std::vector<std::pair<double, double>> validation_scenarios =
        (fixed_grip_training || config.rl.preserve_source_road) ? std::vector<std::pair<double, double>>{
        {fixed_validation_speed(30.0), middle_grip},
        {fixed_validation_speed(35.0), middle_grip},
        {fixed_validation_speed(36.0), middle_grip},
        {fixed_validation_speed(40.0), middle_grip},
        {fixed_validation_speed(45.0), middle_grip},
        {fixed_validation_speed(50.0), middle_grip},
        {fixed_validation_speed(60.0), middle_grip},
        {fixed_validation_speed(70.0), middle_grip},
        {fixed_validation_speed(75.0), middle_grip},
        {fixed_validation_speed(90.0), middle_grip},
        {train_max, middle_grip}} : [&] {
            std::vector<std::pair<double, double>> grid;
            constexpr std::array<double, 6> speed_fractions{{
                0.0, 1.0/9.0, 5.0/18.0, 4.0/9.0, 2.0/3.0, 1.0}};
            constexpr std::array<double, 4> grip_fractions{{
                0.0, 1.0/3.0, 2.0/3.0, 1.0}};
            for (double sf : speed_fractions)
                for (double gf : grip_fractions)
                    grid.emplace_back(train_min + sf * (train_max - train_min),
                        sampled_mu_min + gf * (sampled_mu_max - sampled_mu_min));
            return grid;
        }();
    if (config.rl.focus_target_speeds) {
        for (double kmh : {60.0, 80.0, 95.0, 98.0, 100.0, 102.0, 104.0, 105.0, 116.0, 120.0}) {
            const double speed = kmh / 3.6;
            if (speed >= train_min && speed <= train_max && std::abs(speed - nominal_speed) > 1e-6)
                validation_scenarios.emplace_back(speed, nominal_grip);
        }
    }
    // Identical fixed-speed/unchanged-road cases carry no independent information.
    if (std::abs(train_max - train_min) < 1e-9 &&
        (fixed_grip_training || config.rl.preserve_source_road))
        validation_scenarios.clear();
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
    std::vector<std::pair<double, double>> balanced_cycle_scores;
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
        double rmse_heading_deg;
    };
    std::vector<ValidationCase> best_cases;
    best_cases.push_back({3.6 * nominal_speed, nominal_grip,
        fixed_validation.completed, fixed_validation.rmse,
        fixed_validation.max_abs_error, fixed_validation.mean_reward,
        fixed_validation.grip_limited_steps, fixed_validation.rmse_heading_deg});
    for (std::size_t i = 0; i < validation_scenarios.size(); ++i)
        best_cases.push_back({3.6 * validation_scenarios[i].first,
            validation_scenarios[i].second, fixed_scenarios[i + 1].completed,
            fixed_scenarios[i + 1].rmse, fixed_scenarios[i + 1].max_abs_error,
            fixed_scenarios[i + 1].mean_reward,
            fixed_scenarios[i + 1].grip_limited_steps, fixed_scenarios[i + 1].rmse_heading_deg});
    const auto best_model_file = config.output / "best_validation_policy.txt";
    const auto best_state_file = config.output / "best_training_state.txt";
    std::ostringstream state_context;
    state_context << std::setprecision(17) << "MPC_CONTEXT_V6 "
        << config.rl.preserve_source_road << ' ' << config.rl.vary_road_transition << ' '
        << config.rl.focus_target_speeds << ' ' << config.rl.curriculum << ' '
        << config.mpc.speed_rate_enabled << ' ' << config.mpc.speed_rate_start << ' '
        << config.mpc.speed_rate_end << ' ' << config.mpc.speed_rate_end_limit << ' '
        << config.mpc.nc << ' ' << config.mpc.dt << ' ' << config.mpc.max_steer << ' '
        << config.mpc.max_steer_rate << ' ' << config.mpc.grip_switch_enabled << ' ' << config.mpc.grip_state_only << ' '
        << config.mpc.grip_preview_s << ' ' << config.mpc.nonlinear_tire << ' '
        << config.mpc.tire_slope_floor << ' ' << config.mpc.admm_rho << ' '
        << config.mpc.admm_max_iterations << ' ' << config.mpc.admm_primal_tol << ' '
        << config.mpc.admm_dual_tol << ' ' << config.distance_m << ' '
        << config.carsim.steering_left_index << ' ' << config.carsim.steering_right_index
        << ' ' << config.carsim.steering_gain << ' '
        << std::quoted(config.carsim.solver_dll.generic_string());
    for (const auto& keyword : config.carsim.state_keywords)
        state_context << ' ' << std::quoted(keyword);
    state_context << " warm_sequence_v1";
    // Keep replay from a different speed sampling distribution separate.
    // Older default-range states retain their original context.
    if (std::abs(config.rl.train_speed_min * 3.6 - 30.0) > 1e-9 ||
        std::abs(config.rl.train_speed_max * 3.6 - 120.0) > 1e-9)
        state_context << " speed_range_v1 " << config.rl.train_speed_min
                      << ' ' << config.rl.train_speed_max;
    {
        std::ofstream context_file(config.output / "training_state_context.txt", std::ios::binary);
        context_file << state_context.str();
        if (!context_file) throw std::runtime_error("Cannot save training-state context");
    }
    policy->save(best_model_file); // deterministic actor initially equals fixed MPC
    policy->saveTrainingState(best_state_file);
    bool resume_state = !config.rl.resume_state_file.empty();
    if (!config.rl.warm_start_file.empty() || resume_state) {
        const auto warm_source = std::filesystem::absolute(resume_state ?
            config.rl.resume_state_file : config.rl.warm_start_file);
        if (!std::filesystem::is_regular_file(warm_source))
            throw std::runtime_error("Initialization file does not exist: " + warm_source.string());
        if (resume_state) {
          try {
            std::ifstream old_context(warm_source.parent_path() / "training_state_context.txt",
                                      std::ios::binary);
            const std::string old_context_text{std::istreambuf_iterator<char>(old_context),
                                               std::istreambuf_iterator<char>()};
            if (!old_context || old_context_text != state_context.str())
                throw std::runtime_error("Resume state has incompatible or missing MPC training context; use actor-only warm start");
            std::ifstream old_tires(warm_source.parent_path() / "tire_data.csv", std::ios::binary);
            std::ifstream new_tires(config.output / "tire_data.csv", std::ios::binary);
            if (!old_tires || !new_tires ||
                !std::equal(std::istreambuf_iterator<char>(old_tires),
                            std::istreambuf_iterator<char>(),
                            std::istreambuf_iterator<char>(new_tires),
                            std::istreambuf_iterator<char>()))
                throw std::runtime_error("Resume state tire data differs; use actor-only warm start");
            const auto old_snapshot_file = warm_source.parent_path() / "training_source_Run_all.par";
            std::ifstream old_source(old_snapshot_file, std::ios::binary);
            if (!old_source) throw std::runtime_error("Resume state requires its training_source_Run_all.par");
            const std::string old_snapshot{std::istreambuf_iterator<char>(old_source),
                                           std::istreambuf_iterator<char>()};
            const auto canonical_source = [](const std::string& source) {
                std::istringstream in(source);
                std::ostringstream out;
                std::string line;
                while (std::getline(in, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    std::istringstream fields(line);
                    std::string keyword;
                    fields >> keyword;
                    if (keyword == "SPEED_TARGET_CONSTANT" || keyword == "MU_ROAD_CONSTANT" ||
                        keyword == "TSTOP") out << keyword << " <variable>\n";
                    else out << line << '\n';
                }
                return out.str();
            };
            if (canonical_source(old_snapshot) != canonical_source(source_snapshot))
                throw std::runtime_error("Resume source changed beyond speed, grip or stop time; use actor-only warm start for a different vehicle/road");
            policy->loadTrainingState(warm_source);
          } catch (const std::exception& e) {
            if (!config.rl.resume_actor_fallback) throw;
            std::cerr << "Training context changed (" << e.what()
                      << "); restoring Actor only, resetting Critic and replay.\n";
            auto old_policy = createRlPolicy(config.rl);
            old_policy->loadTrainingState(warm_source);
            const auto actor_file = config.output / "warm_start_input_policy.txt";
            old_policy->save(actor_file);
            policy = createRlPolicy(config.rl);
            policy->warmStart(actor_file);
            resume_state = false;
          }
        } else {
            std::filesystem::copy_file(warm_source, config.output / "warm_start_input_policy.txt",
                                       std::filesystem::copy_options::overwrite_existing);
            policy->warmStart(warm_source);
        }
        policy->setTrainingEpisode(1, config.rl.episodes);
        policy->save(config.output / "warm_start_rebased_policy.txt");
        std::vector<ValidationCase> warm_cases;
        double warm_score = 0.0;
        int warm_completed = 0;
        bool warm_acceptable = true;
        std::ofstream warm_log(config.output / "warm_start_validation.csv", std::ios::binary);
        if (!warm_log) throw std::runtime_error("Cannot write warm-start validation");
        warm_log << "speed_kmh,mu,completed,rmse_y_m,max_abs_y_m,mean_step_reward,rmse_heading_deg,grip_limited_steps\n";
        for (std::size_t i = 0; i < fixed_scenarios.size(); ++i) {
            const double speed = i == 0 ? nominal_speed : validation_scenarios[i - 1].first;
            const double grip = i == 0 ? nominal_grip : validation_scenarios[i - 1].second;
            const auto candidate = runEpisode(config, speed, policy.get(), false, grip);
            verify_source();
            const auto& baseline = fixed_scenarios[i];
            if (config.rl.penalize_qp_failure)
                warm_acceptable &= candidate.fallback_steps <= baseline.fallback_steps;
            if (config.rl.strict_tracking_selection)
                warm_acceptable &= trackingSafetyAcceptable(candidate.rmse_heading_deg, candidate.grip_limited_steps,
                    baseline.rmse_heading_deg, baseline.grip_limited_steps);
            const bool low_speed = 3.6 * speed <= 45.0 + 1e-6;
            if (i == 0) {
                warm_acceptable &= candidate.rmse <= baseline.rmse * 1.05 + 0.02;
                warm_acceptable &= candidate.max_abs_error <= baseline.max_abs_error * 1.10 + 0.05;
            } else {
                warm_acceptable &= candidate.rmse <= baseline.rmse * (low_speed ? 1.05 : 1.10) +
                    (low_speed ? 0.005 : 0.05);
                warm_acceptable &= candidate.max_abs_error <= baseline.max_abs_error * (low_speed ? 1.02 : 1.10) +
                    (low_speed ? 0.005 : 0.05);
            }
            if (config.rl.action_profile != "full" && isPrioritySpeed(speed))
                warm_acceptable &= candidate.rmse <= baseline.rmse;
            warm_score += candidate.mean_reward;
            warm_completed += candidate.completed ? 1 : 0;
            warm_cases.push_back({3.6 * speed, grip, candidate.completed, candidate.rmse,
                candidate.max_abs_error, candidate.mean_reward, candidate.grip_limited_steps, candidate.rmse_heading_deg});
            warm_log << 3.6 * speed << ',' << grip << ',' << candidate.completed << ','
                     << candidate.rmse << ',' << candidate.max_abs_error << ','
                     << candidate.mean_reward << ',' << candidate.rmse_heading_deg << ','
                     << candidate.grip_limited_steps << '\n';
        }
        warm_score /= fixed_scenarios.size();
        if (warm_acceptable && warm_completed >= fixed_completed_scenarios &&
            warm_score > best_selection_score + 0.01) {
            best_selection_score = warm_score;
            best_completed_scenarios = warm_completed;
            best_episode = -1;
            best_cases = std::move(warm_cases);
            policy->save(best_model_file);
            policy->saveTrainingState(best_state_file);
        }
        std::ofstream provenance(config.output / "warm_start_manifest.txt", std::ios::binary);
        provenance << "source=" << warm_source.string() << '\n'
                   << "initialization=" << (resume_state ? "full_training_state" : "actor_only") << '\n'
                   << "critic_optimizer_and_replay_restored=" << (resume_state ? "true" : "false") << '\n'
                   << "initial_mean_reward=" << warm_score << '\n'
                   << "initial_acceptable=" << warm_acceptable << '\n'
                   << "initial_selected=" << (best_episode == -1) << '\n';
        std::cout << (resume_state ? "Resume training: " : "Actor warm start: ") << warm_source.string()
                  << "; initial score=" << warm_score
                  << "; accepted=" << (best_episode == -1) << '\n';
    }
    const double initial_exploration = policy->explorationStdDev();
    std::ofstream log(config.output / "training_rewards.csv", std::ios::binary);
    if (!log) throw std::runtime_error("Cannot write training_rewards.csv");
    log << "episode,speed_kmh,mu,episode_return,episode_return_10,"
           "validation_return,validation_return_10,rl_reference_return,"
           "validation_advantage,validation_advantage_10,"
           "mean_step_reward,rmse_y_m,max_abs_y_m,completed,"
           "paired_rl_reference_mean_step_reward,paired_advantage_per_step,"
           "paired_advantage_10,exploration_sigma,road_station_scale\n";
    std::ofstream checkpoint_log(config.output / "checkpoint_summary.csv", std::ios::binary);
    if (!checkpoint_log) throw std::runtime_error("Cannot write checkpoint_summary.csv");
    checkpoint_log << "episode,mean_reward_advantage_per_step,accepted,"
                      "completed_scenarios,acceptable_errors\n";
    std::ofstream scenario_log(config.output / "checkpoint_validation.csv", std::ios::binary);
    if (!scenario_log) throw std::runtime_error("Cannot write checkpoint_validation.csv");
    scenario_log << "episode,speed_kmh,mu,completed,rmse_y_m,max_abs_y_m,mean_step_reward,"
                    "episode_return,grip_limited_steps,fallback_steps,rmse_heading_deg\n";
    std::ofstream cycle_log(config.output / "training_cycle_summary.csv", std::ios::binary);
    if (!cycle_log) throw std::runtime_error("Cannot write training_cycle_summary.csv");
    cycle_log << "last_episode,matched_mean_reward_advantage_per_step\n";
    std::ofstream optimization_log(config.output / "optimization_metrics.csv", std::ios::binary);
    if (!optimization_log) throw std::runtime_error("Cannot write optimization_metrics.csv");
    optimization_log << "episode,critic1_huber_loss,critic2_huber_loss,actor_surrogate_objective,critic_updates,actor_updates,transitions,clipped_rewards\n";
    {
        std::ofstream steps(config.output / "diagnostic_steps.csv");
        std::ofstream decisions(config.output / "diagnostic_decisions.csv");
        if (!steps || !decisions) throw std::runtime_error("Cannot create training diagnostics");
        steps << "episode,step,station_m,ey_m,epsi_rad,step_reward,loss_clipped,qp_failure,grip_limited,cost_y,cost_peak,cost_psi,cost_steer,cost_steer_rate,cost_np,cost_vy,cost_yaw,cost_weight_change,cost_weight_deviation,cost_grip\n";
        decisions << "episode,block,station_m,block_mean,baseline_mean,training_reward,reward_clipped,done";
        for (int i = 0; i < 17; ++i) decisions << ",state_" << i;
        decisions << ",np,q_y,q_psi,r_delta\n";
    }
    std::vector<std::pair<double, double>> critic1_losses, critic2_losses, actor_objectives;
    policy->takeTrainingStats();
    for (std::size_t i = 0; i < fixed_scenarios.size(); ++i) {
        const double case_speed = i == 0 ? nominal_speed : validation_scenarios[i - 1].first;
        const double case_grip = i == 0 ? nominal_grip : validation_scenarios[i - 1].second;
        const auto& fixed = fixed_scenarios[i];
        scenario_log << 0 << ',' << 3.6 * case_speed << ',' << case_grip << ','
                     << fixed.completed << ',' << fixed.rmse << ','
                     << fixed.max_abs_error << ',' << fixed.mean_reward
                     << ',' << fixed.reward_sum << ',' << fixed.grip_limited_steps
                     << ',' << fixed.fallback_steps << ',' << fixed.rmse_heading_deg << '\n';
    }
    for (int episode = 1; episode <= config.rl.episodes; ++episode) {
        const auto [speed_bin, grip_bin] = scenario_bins[episode - 1];
        const double speed_fraction = speed_bin_fraction[speed_bin] +
            unit_distribution(rng) *
            (speed_bin_fraction[speed_bin + 1] - speed_bin_fraction[speed_bin]);
        const double speed = train_min + speed_fraction * (train_max - train_min);
        const double grip = config.rl.preserve_source_road ? nominal_grip : fixed_grip_training ? config.rl.train_mu_min :
            sampled_mu_min +
            (grip_bin + unit_distribution(rng)) / grip_bins *
            (sampled_mu_max - sampled_mu_min);
        const double road_scale = config.rl.vary_road_transition ?
            0.85 + 0.30 * unit_distribution(rng) : 1.0;
        policy->setTrainingEpisode(episode, config.rl.episodes);
        const auto paired_fixed = runEpisode(config, speed, nullptr, false, grip, nullptr, road_scale);
        verify_source();
        const auto run = runEpisode(config, speed, policy.get(), true, grip,
                                    &paired_fixed.block_mean_rewards, road_scale, episode);
        const auto optimization = policy->takeTrainingStats();
        optimization_log << episode << ',';
        if (optimization.critic_updates > 0) {
            const double loss = optimization.critic1_loss_sum / optimization.critic_updates;
            optimization_log << loss;
            critic1_losses.emplace_back(episode, loss);
        }
        optimization_log << ',';
        if (config.rl.algorithm == "td3" && optimization.critic_updates > 0) {
            const double loss = optimization.critic2_loss_sum / optimization.critic_updates;
            optimization_log << loss;
            critic2_losses.emplace_back(episode, loss);
        }
        optimization_log << ',';
        if (optimization.actor_updates > 0) {
            const double objective = optimization.actor_objective_sum / optimization.actor_updates;
            optimization_log << objective;
            actor_objectives.emplace_back(episode, objective);
        }
        optimization_log << ',' << optimization.critic_updates << ',' << optimization.actor_updates
                         << ',' << optimization.transitions << ',' << optimization.clipped_rewards << '\n';
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
        if (episode % cycle_size == 0) {
            double cycle_mean = 0.0;
            for (int k = episode - cycle_size; k < episode; ++k)
                cycle_mean += paired_advantages[static_cast<std::size_t>(k)].second;
            cycle_mean /= cycle_size;
            balanced_cycle_scores.emplace_back(episode, cycle_mean);
            cycle_log << episode << ',' << cycle_mean << '\n';
        }
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
            std::ostringstream checkpoint_name;
            checkpoint_name << "checkpoint_" << std::setfill('0') << std::setw(4)
                            << episode << "_policy.txt";
            policy->save(config.output / checkpoint_name.str());
            double score = validation.mean_reward;
            int completed_scenarios = validation.completed ? 1 : 0;
            bool acceptable_errors = validation.rmse <=
                fixed_validation.rmse * 1.05 + (config.rl.action_profile == "np_only" ? 0.005 : 0.02) &&
                validation.max_abs_error <=
                fixed_validation.max_abs_error * 1.10 + 0.05;
            if (config.rl.penalize_qp_failure)
                acceptable_errors &= validation.fallback_steps <= fixed_validation.fallback_steps;
            if (config.rl.strict_tracking_selection)
                acceptable_errors &= trackingSafetyAcceptable(validation.rmse_heading_deg, validation.grip_limited_steps,
                    fixed_validation.rmse_heading_deg, fixed_validation.grip_limited_steps);
            if (config.rl.action_profile != "full" && isPrioritySpeed(nominal_speed))
                acceptable_errors &= validation.rmse <= fixed_validation.rmse;
            std::vector<ValidationCase> checked_cases{{
                3.6 * nominal_speed, nominal_grip, validation.completed,
                validation.rmse, validation.max_abs_error,
                validation.mean_reward, validation.grip_limited_steps, validation.rmse_heading_deg}};
            scenario_log << episode << ',' << 3.6 * nominal_speed << ',' << nominal_grip
                         << ',' << validation.completed << ',' << validation.rmse << ','
                         << validation.max_abs_error << ','
                         << validation.mean_reward << ',' << validation.reward_sum << ','
                         << validation.grip_limited_steps << ','
                         << validation.fallback_steps << ',' << validation.rmse_heading_deg << '\n';
                std::size_t scenario_index = 1;
                for (const auto [case_speed, case_grip] : validation_scenarios) {
                    const auto check = runEpisode(config, case_speed, policy.get(),
                                                  false, case_grip);
                    verify_source();
                    score += check.mean_reward;
                    completed_scenarios += check.completed ? 1 : 0;
                    const auto& baseline = fixed_scenarios[scenario_index++];
                    if (config.rl.penalize_qp_failure)
                        acceptable_errors &= check.fallback_steps <= baseline.fallback_steps;
                    if (config.rl.strict_tracking_selection)
                        acceptable_errors &= trackingSafetyAcceptable(check.rmse_heading_deg, check.grip_limited_steps,
                            baseline.rmse_heading_deg, baseline.grip_limited_steps);
                    const bool low_speed = 3.6 * case_speed <= 45.0 + 1e-6;
                    const bool target_speed = config.rl.action_profile != "full" && isPrioritySpeed(case_speed);
                    acceptable_errors &= check.rmse <=
                        baseline.rmse * (low_speed || target_speed ? 1.05 : 1.10) +
                        (low_speed || target_speed ? 0.005 : 0.05);
                    if (target_speed) acceptable_errors &= check.rmse <= baseline.rmse;
                    acceptable_errors &= check.max_abs_error <=
                        baseline.max_abs_error * (low_speed ? 1.02 : 1.10) +
                        (low_speed ? 0.005 : 0.05);
                    checked_cases.push_back({3.6 * case_speed, case_grip,
                        check.completed, check.rmse, check.max_abs_error,
                        check.mean_reward, check.grip_limited_steps, check.rmse_heading_deg});
                    scenario_log << episode << ',' << 3.6 * case_speed << ','
                                 << case_grip << ',' << check.completed << ','
                                 << check.rmse << ',' << check.max_abs_error << ','
                                 << check.mean_reward << ','
                                 << check.reward_sum << ',' << check.grip_limited_steps
                                 << ',' << check.fallback_steps << ',' << check.rmse_heading_deg << '\n';
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
                policy->saveTrainingState(best_state_file);
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
            << policy->explorationStdDev() << ',' << road_scale << '\n';
        std::cout << algorithm_name << " episode " << episode << '/' << config.rl.episodes
                  << " speed=" << 3.6 * speed << " km/h mu=" << grip
                  << " mean_reward=" << run.mean_reward
                  << " RMSE=" << run.rmse << " completed=" << run.completed << '\n';
    }
    log.flush();
    policy->save(config.output / "last_unvalidated_policy.txt");
    policy->saveTrainingState(config.output / "last_training_state.txt");
    policy->load(best_model_file);
    if (best_episode == 0)
        std::cerr << "WARNING: No learned checkpoint beat the RL-neutral reference "
                     "without excessive tracking error; preserving the neutral policy.\n";
    policy->save(config.output / config.rl.model_file);
    {
        std::ofstream selected_log(config.output / "selected_validation.csv", std::ios::binary);
        if (!selected_log) throw std::runtime_error("Cannot write selected_validation.csv");
        selected_log << "seed,checkpoint_episode,controller,speed_kmh,mu,completed,"
                        "rmse_y_m,max_abs_y_m,mean_step_reward,grip_limited_steps,rmse_heading_deg\n";
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
                         << fixed.grip_limited_steps << ',' << fixed.rmse_heading_deg << '\n';
            selected_log << config.rl.seed << ',' << best_episode << ','
                         << config.rl.algorithm << ','
                         << learned.speed_kmh << ',' << learned.mu << ','
                         << learned.completed << ',' << learned.rmse_m << ','
                         << learned.max_abs_error_m << ',' << learned.mean_reward << ','
                         << learned.grip_limited_steps << ',' << learned.rmse_heading_deg << '\n';
        }
    }
    chart(config.output / "training_rewards.svg", algorithm_name + " training episode return: 10-episode mean",
          "Episode", "Episode return (reward points)",
          {mean_rewards}, {"10-episode mean"}, false, true);
    if (config.rl.algorithm == "td3")
        chart(config.output / "critic_losses.svg", "Critic minibatch Huber losses",
              "Episode", "Huber loss", {critic1_losses, critic2_losses}, {"Critic 1", "Critic 2"}, false, true);
    else
        chart(config.output / "critic_losses.svg", "Critic minibatch Huber loss",
              "Episode", "Huber loss", {critic1_losses}, {"Critic 1"}, false, true);
    chart(config.output / "actor_objective.svg", "Actor surrogate objective: -Q plus action penalty",
          "Episode", "Surrogate objective", {actor_objectives}, {"Actor objective"}, false, true);
    chart(config.output / "training_episodes.svg", algorithm_name + " exploratory episode returns",
          "Episode", "Episode return (reward points)",
          {rewards, mean_rewards}, {"Episode", "10-episode mean"}, false, true);
    chart(config.output / "paired_advantage.svg",
          "Training reward advantage over RL-neutral MPC at matched speed and grip",
          "Episode", "Reward advantage per step",
          {paired_advantages, paired_means, zero_reference},
          {"Paired advantage", "10-episode mean", "RL reference"}, false, true);
    chart(config.output / "training_cycle_convergence.svg",
          "Balanced training cycles: matched reward advantage",
          "Last episode in cycle", "Mean reward advantage per step",
          {balanced_cycle_scores}, {"Complete cycle mean"}, false, true);
    chart(config.output / "checkpoint_convergence.svg",
          "Deterministic multi-scenario validation reward advantage",
          "Episode", "Mean reward advantage per step",
          {checkpoint_scores, checkpoint_best, checkpoint_zero},
          {"Current policy", "Best admissible", "RL reference"}, false, true);
    chart(config.output / "validation_returns.svg", algorithm_name + " validation return at fixed speed",
          "Episode", "Episode return (reward points)",
          {validation_rewards, validation_means, fixed_reference},
          {algorithm_name, "10-episode mean", "RL reference"}, false, true);
    chart(config.output / "validation_advantage.svg", algorithm_name + " return minus RL reference",
          "Episode", "Return difference (reward points)",
          {advantages, advantage_means, zero_reference},
          {algorithm_name + " advantage", "10-episode mean", "Equal return"}, false, true);
    if (best_episode > 0)
        std::cout << "Best validation checkpoint: episode " << best_episode
                  << ", score=" << best_selection_score << '\n';
    else if (best_episode == -1)
        std::cout << "Best validation checkpoint: initialization actor, score="
                  << best_selection_score << '\n';
    else
        std::cout << "Best validation checkpoint: RL-neutral reference (episode 0)\n";

        const auto selected = runEpisode(config, nominal_speed, policy.get(),
                                         false, nominal_grip);
        std::ofstream report(config.output / "rl_report.html", std::ios::binary);
        report << "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
               << "<title>CarSim " << algorithm_name << " training</title><style>body{font:16px Arial;max-width:1050px;margin:30px auto;color:#172033}img{width:100%;border:1px solid #ddd;margin:10px 0}table{border-collapse:collapse;width:100%}th,td{border:1px solid #cbd5e1;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}</style>"
               << "<h1>CarSim " << algorithm_name << (config.rl.action_profile == "np_only" ?
                    ": 自适应 Np，固定 Q 和 R</h1>" : ": 自适应 Np、Q 和 R</h1>")
               << "<p>MPC 仅使用六状态三自由度动力学 [vy,vx,psi,r,Y,X]，沿预测时域逐步线性化、"
                  "Tustin 离散化，并以转角增量为优化量。纵向轮胎滑移与刚度未映射，"
                  "由相邻 CarSim 状态的纵向加速度表示当前纵向作用。"
               << algorithm_name << (config.rl.action_profile == "np_only" ?
                    " 在全部速度范围调节 Np，Q/R 固定。</p>" : " 在全部速度范围调节 Np/Q/R。</p>")
               << (config.mpc.nonlinear_tire ?
                   "<p>实验模型：沿名义轨迹对非线性轮胎表局部线性化，低斜率采用下限保护；此前线性模型经验不能直接混用。</p>" : "")
               << "<p>训练环境：CarSim VS Solver；车辆、道路和参考轨迹来自指定 Simfile 的 Parsfile。"
               << (config.rl.preserve_source_road ?
                   "每回合在私有副本设置目标车速和必要的 TSTOP，保留原道路附着表；原工况不修改。训练回合：" :
                   "每回合在私有副本设置目标车速、附着（常量或按原分段比例缩放的阶梯表）和必要的 TSTOP；原工况不修改。训练回合：")
               << config.rl.episodes << "，随机种子：" << config.rl.seed
               << "，速度范围：" << 3.6 * train_min << "–" << 3.6 * train_max
               << " km/h；路面附着系数 μ 范围：" << config.rl.train_mu_min
               << "–" << config.rl.train_mu_max
               << "；训练采样的初始 μ：" << sampled_mu_min << "–" << sampled_mu_max
               << (config.rl.preserve_source_road ? "（原道路初始值）" : "（变附着时按全表范围约束）")
               << "；基准验证：" << 3.6 * nominal_speed << " km/h、μ="
               << nominal_grip << "。速度采用 " << speed_bins << " 档分层抽样；"
               << (config.rl.focus_target_speeds ? "按 30–120 km/h 范围加密 55–65、75–85、95–105 km/h；" : "低速区间加密；")
               << (config.rl.preserve_source_road ? "保留原道路；" : fixed_grip_training ? "μ 固定；" :
                   "μ 分 5 档；")
               << "当前位置与前方曲率从 CarSim 参考轨迹逐步计算。"
               << (config.rl.vary_road_transition ? "训练时道路表站距按 0.85–1.15 随机缩放，配对参考使用相同道路；验证保持原表。" : "")
               << "曲率沿该轨迹变化，训练未随机生成其他道路轨迹。</p>"
               << "<p>动作：Np " << config.rl.min_np << "–" << config.rl.max_np
               << "，中性值 " << config.rl.nominal_np << "；Qy "
               << config.rl.weight_min[0] << "–" << config.rl.weight_max[0]
               << "；Qpsi " << config.rl.weight_min[1] << "–" << config.rl.weight_max[1]
               << "；Qvy " << config.rl.weight_min[2] << "–" << config.rl.weight_max[2]
               << "；Qr " << config.rl.weight_min[3] << "–" << config.rl.weight_max[3]
               << "；Rsteer " << config.rl.weight_min[4] << "–" << config.rl.weight_max[4]
               << "；Rdelta " << config.rl.weight_min[5] << "–" << config.rl.weight_max[5]
               << "。Q/R 是绝对数值，与固定 MPC 的 [weights] 无关。Rsteer 约束重构后的转角与曲率前馈之差，"
                  "Rdelta 约束优化变量本身，即转角增量。"
               << algorithm_name << " 在所有速度运行；"
               << (config.rl.action_profile == "np_only" ?
                   "仅训练 Np；Q/R 保持各自范围的几何中心。" : config.rl.action_profile == "compact" ?
                   "精简动作只训练 Np、Qy、Qpsi、Rdelta，Qvy、Qr、Rsteer 固定在区间几何中心；" :
                   "完整动作训练 Np 与六个 Q/R 权重；")
               << (config.rl.low_speed_blend ?
                   "低速动作混合已启用：起点及以下使用中性 MPC，终点及以上使用完整策略，中间线性过渡。" : "")
               << "整数 Np 在小幅动作下可能不变；模型、动作范围和策略格式必须匹配。</p>"
               << "<p>奖励是横向及航向误差、侧向速度、横摆速率误差、转角、转角变化、"
               << "预测时域长度与跳变、Q/R 偏离及跳变、受限控制的加权负代价；"
               << "求解失败和未完成工况另加惩罚。每次训练也运行同车速、同附着的中性参考 MPC。"
               << (config.rl.direct_block_reward ?
                  "Critic 使用决策块平均原始回报除以 reward_scale，并截断至 ±8；检查点仍按整段回报和安全门槛选择。" :
                  (config.rl.relative_advantage ?
                   "Critic 使用决策块回报差，按 max(0.2×reward_scale, |基准回报|) 归一化并截断至 ±8。" :
                   "Critic 使用决策块回报差除以 reward_scale 后截断至 ±8。"))
               << (config.rl.align_integer_np ? "Critic 前向使用整数化的 Np，Actor 对 Np 使用直通近似梯度。" : "")
               << "探索噪声标准差逐回合从 " << initial_exploration
               << " 降至 0.04。Critic 用 64 样本平均 Huber 损失（阈值 2），Actor 用 32 样本更新一次；"
               << "每 2 次 Critic 更新执行一次 Actor 与目标网络更新，"
               << "并加入按横向加速度需求调整的动作平方正则；"
               << (config.rl.algorithm == "td3" ?
                   "TD3 使用双 Critic 的较小目标值与有界目标动作噪声。" :
                   "DDPG 保留单 Critic，不使用目标动作噪声。")
               << (config.rl.curriculum && config.rl.preserve_source_road ?
                   "每个周期按速度从低到高推进。" : config.rl.curriculum ?
                   "训练顺序采用逐周期课程：每个周期按速度和初始附着分层由低到高推进，"
                   "同一层随机排序；保留低附着高速及高附着低速交叉工况。" :
                   "训练工况在每个分层周期内随机排序。")
               << "每 " << cycle_size << " 回合完整覆盖一次训练分层网格，"
                  "其配对优势均值比随机单回合回报更适合观察训练趋势。</p>"
               << "<p>每步奖励 = −" << config.rl.reward.reward_scale
               << " × min(120, 加权代价)。跟踪项：" << config.rl.reward.w_y
               << "·Ly(ey/" << config.rl.reward.ey_scale_m
               << ") + " << config.rl.reward.w_peak
               << "·[max(0,|ey|−" << config.rl.reward.peak_threshold_m
               << ")/" << config.rl.reward.peak_threshold_m << "]²"
               << " + " << config.rl.reward.w_psi << "·L(epsi/"
               << config.rl.reward.epsi_scale_rad
               << ")；横向和航向误差均用 L(z)=(√(1+z²)−1)/(√2−1)，"
                  "输入限于 ±20。侧向速度与横摆误差："
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
               << (best_episode == -1 ? "继承的初始策略" :
                   (best_episode ? std::to_string(best_episode) : "0；保留 RL 中性策略"))
               << "；该检查点完成 " << std::max(0, best_completed_scenarios)
               << "/" << fixed_scenarios.size() << " 个速度-附着验证工况。"
               << "完成整段仿真不代表轨迹误差达标；以下逐项列出横向 RMSE 和最大误差。"
               << "这些指标仅为本次车辆和轨迹验证，不代表其他工况的泛化性能。</p>"
               << (config.rl.strict_tracking_selection ?
                   "<p>选优同时要求航向 RMSE 不超过对应基准的 105% + 0.02°，附着保护步数不超过对应基准。</p>" : "")
               << "<h2>选中检查点的无探索验证</h2>"
               << "<table><tr><th>目标车速 (km/h)</th><th>路面 μ</th><th>完成</th>"
                  "<th>横向 RMSE (m)</th><th>最大横向误差 (m)</th><th>航向 RMSE (°)</th><th>受限周期</th></tr>";
        for (const auto& check : best_cases)
            report << "<tr><td>" << check.speed_kmh << "</td><td>" << check.mu
                   << "</td><td>" << (check.completed ? "是" : "否")
                   << "</td><td>" << check.rmse_m << "</td><td>"
                   << check.max_abs_error_m << "</td><td>" << check.rmse_heading_deg << "</td><td>"
                   << check.grip_limited_steps << "</td></tr>";
        report << "</table><p>表中速度是 CarSim 的目标设定值；实际车速应查看对应仿真的"
                  " simulation.csv 或求解器输出。基准工况以外的验证工况用于检查点选择，"
                  "未向策略提供探索噪声。</p>"
               << "<p><a href=\"training_rewards.csv\">逐回合数据</a> · "
               << "<a href=\"optimization_metrics.csv\">Critic 损失和 Actor 代理目标</a> · "
               << "<a href=\"training_cycle_summary.csv\">完整分层周期均值</a> · "
               << "<a href=\"effective_config.ini\">生效配置</a> · "
               << "<a href=\"training_source_Run_all.par\">工况原始快照</a> · "
               << "<a href=\"checkpoint_validation.csv\">速度-附着验证明细</a> · "
               << "<a href=\"checkpoint_summary.csv\">固定工况收敛记录</a> · "
               << "<a href=\"selected_validation.csv\">选中策略与 RL 中性参考逐工况对照</a> · "
               << "<a href=\"best_training_state.txt\">选中检查点完整训练状态</a> · "
               << "<a href=\"last_training_state.txt\">末次完整训练状态</a> · "
               << "<a href=\"" << config.rl.model_file.generic_string()
               << "\">策略文件</a></p>"
               << "<p>随机训练回合工况不同，原始回报曲线不能直接判断收敛。"
                  "下图先显示固定验证工况的无探索回报优势与完整分层周期均值，再显示同工况配对优势；"
                  "固定工况图同时给出已通过误差约束的历史最佳值。历史最佳曲线单调不代表当前策略收敛。</p>"
               << "<p>Critic 为训练批次 Huber 损失；Actor 为 -Q 加动作惩罚的代理目标。这些曲线不能单独证明闭环收敛。</p>"
               << "<img src=\"critic_losses.svg\"><img src=\"actor_objective.svg\">"
               << "<img src=\"checkpoint_convergence.svg\"><img src=\"training_cycle_convergence.svg\">"
               << "<img src=\"paired_advantage.svg\">"
               << "<img src=\"validation_returns.svg\"><img src=\"training_episodes.svg\">"
               << "<img src=\"validation_advantage.svg\"></html>";
        std::cout << "CarSim training report: "
                  << std::filesystem::absolute(config.output / "rl_report.html").string() << '\n';
}

void exportRlTrainingActor(const AppConfig& config) {
    if (config.rl.resume_state_file.empty())
        throw std::invalid_argument("export_state requires rl.resume_state_file");
    const auto state_file = std::filesystem::absolute(config.rl.resume_state_file);
    if (!std::filesystem::is_regular_file(state_file))
        throw std::runtime_error("Training state does not exist: " + state_file.string());
    const auto source_dir = state_file.parent_path();
    const auto source_snapshot = source_dir / "training_source_Run_all.par";
    if (!std::filesystem::is_regular_file(source_snapshot))
        throw std::runtime_error("Training-state source snapshot is missing");
    auto policy = createRlPolicy(config.rl);
    policy->loadTrainingState(state_file);
    std::filesystem::create_directories(config.output);
    const auto policy_file = config.output /
        (config.rl.algorithm == "td3" ? "td3_policy.txt" : "ddpg_policy.txt");
    policy->save(policy_file);
    std::filesystem::copy_file(source_snapshot, config.output / "training_source_Run_all.par");
    for (const char* name : {"training_state_context.txt", "tire_data.csv"}) {
        if (std::filesystem::is_regular_file(source_dir / name))
            std::filesystem::copy_file(source_dir / name, config.output / name);
    }
    std::ofstream manifest(config.output / "candidate_manifest.txt", std::ios::binary);
    manifest << "source_training_state=" << state_file.string() << '\n'
             << "status=unvalidated_state_actor\n"
             << "note=not_the_selected_policy_unless_independently_verified\n";
    if (!manifest) throw std::runtime_error("Cannot write exported candidate manifest");
    std::ofstream effective(config.output / "effective_config.ini", std::ios::binary);
    effective << configToIni(config);
    if (!effective) throw std::runtime_error("Cannot write exported candidate config");
    std::cout << "Unvalidated state actor: "
              << std::filesystem::absolute(policy_file).string() << '\n';
}

void runRlHeldoutEvaluation(const AppConfig& input_config) {
    AppConfig config = input_config;
    AppConfig basic_config = input_config;
    const bool focus_evaluation = config.rl.mode == "eval_focus";
    const bool general_evaluation = config.rl.mode == "eval_general" || focus_evaluation;
    if (general_evaluation && !focus_evaluation &&
        (config.rl.train_mu_min == config.rl.train_mu_max ||
         config.rl.train_speed_min == config.rl.train_speed_max))
        throw std::invalid_argument("eval_general needs nonzero speed and mu ranges");
    if (focus_evaluation &&
        (!std::isfinite(config.evaluation_focus_mu) ||
         config.evaluation_focus_mu < config.rl.train_mu_min - 0.005 ||
         config.evaluation_focus_mu > config.rl.train_mu_max + 0.005))
        throw std::invalid_argument("eval_focus requires --focus-mu within the policy training grip range");
    // Match the RL-owned neutral comparator used for training checkpoints.
    config.mpc = applyRlAction(config.mpc, neutralRlAction(config.rl), config.rl);
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
    std::string eval_source_snapshot;
    const auto same_source = [&] {
        if (general_evaluation) {
            std::ifstream in(config.output / "training_solver" / "Run_visualizer.par",
                             std::ios::binary);
            if (!in) return false;
            const std::string current{std::istreambuf_iterator<char>(in),
                                      std::istreambuf_iterator<char>()};
            if (eval_source_snapshot.empty()) {
                eval_source_snapshot = current;
                std::ofstream snapshot(config.output / "evaluation_source_Run_all.par",
                                       std::ios::binary);
                snapshot << current;
                if (!snapshot) throw std::runtime_error("Cannot archive evaluation source");
                std::ifstream training(training_source, std::ios::binary);
                const std::string original{std::istreambuf_iterator<char>(training),
                                           std::istreambuf_iterator<char>()};
                std::ofstream manifest(config.output / "evaluation_manifest.txt");
                manifest << "training_source=" << training_source.string() << '\n'
                         << "evaluation_source_equals_training_source="
                         << (original == current) << '\n';
            }
            return !current.empty() && current == eval_source_snapshot;
        }
        std::ifstream a(training_source, std::ios::binary);
        std::ifstream b(config.output / "training_solver" / "Run_visualizer.par",
                        std::ios::binary);
        return a && b && std::equal(std::istreambuf_iterator<char>(a),
                                    std::istreambuf_iterator<char>(),
                                    std::istreambuf_iterator<char>(b),
                                    std::istreambuf_iterator<char>());
    };
    std::vector<std::pair<double, double>> cases{
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
        {90.0, 0.55}, {90.0, 0.65}, {120.0, 0.6}};
    if (focus_evaluation) {
        cases.clear();
        constexpr std::array<double, 10> focus_speeds{{
            33.0, 37.0, 39.0, 41.0, 46.0, 48.0, 53.0, 57.0, 63.0, 67.0}};
        for (const double speed : focus_speeds)
            if (speed >= 3.6 * config.rl.train_speed_min - 1e-9 &&
                speed <= 3.6 * config.rl.train_speed_max + 1e-9)
                cases.emplace_back(speed, config.evaluation_focus_mu);
        if (cases.empty()) throw std::invalid_argument("No focus speeds in configured range");
    } else if (general_evaluation) {
        cases.clear();
        constexpr std::array<double, 9> speed_fractions{{
            2.0/90.0, 8.0/90.0, 14.0/90.0, 22.0/90.0, 32.0/90.0,
            44.0/90.0, 56.0/90.0, 73.0/90.0, 84.0/90.0}};
        constexpr std::array<double, 4> grip_fractions{{
            1.0/12.0, 1.0/3.0, 2.0/3.0, 11.0/12.0}};
        for (const double gf : grip_fractions)
            for (const double sf : speed_fractions)
                cases.emplace_back(3.6 * (config.rl.train_speed_min +
                    sf * (config.rl.train_speed_max - config.rl.train_speed_min)),
                    config.rl.train_mu_min + gf *
                    (config.rl.train_mu_max - config.rl.train_mu_min));
    }
    const auto output_file = config.output /
        (general_evaluation ? "generalization_validation.csv" : "heldout_validation.csv");
    std::ofstream out(output_file, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write held-out validation CSV");
    out << "speed_kmh,mu,controller,completed,rmse_y_m,max_abs_y_m,"
           "mean_step_reward,grip_limited_steps,fallback_steps,"
           "np_min,np_max,np_unique,policy_decisions,changed_decisions,"
           "max_abs_normalized_weight_deviation\n";
    out << std::setprecision(12);
    std::ofstream demand_log;
    double peak_reference_curvature = -1.0;
    if (general_evaluation) {
        demand_log.open(config.output / "reference_demand.csv", std::ios::binary);
        if (!demand_log) throw std::runtime_error("Cannot write reference demand");
        demand_log << "speed_kmh,mu,peak_abs_curvature_per_m,ideal_peak_ay_mps2,"
                      "ideal_mu_g_mps2,ideal_demand_ratio,ideal_speed_limit_kmh\n"
                   << std::setprecision(12);
    }
    struct Aggregate {
        int cases = 0;
        int completed = 0;
        double rmse_sum = 0.0;
        double worst_error = 0.0;
        double reward_sum = 0.0;
    };
    std::map<std::pair<double, std::string>, Aggregate> aggregates;
    struct GridMetric { double rmse = 0.0, maximum = 0.0; bool completed = false; };
    std::map<std::tuple<double, double, std::string>, GridMetric> grid_metrics;
    std::set<double> evaluated_speeds;
    const bool fixed_grip_evaluation =
        config.rl.train_mu_min == config.rl.train_mu_max || config.rl.preserve_source_road;
    std::vector<double> selection_speeds_kmh;
    if (fixed_grip_evaluation) {
        const double min_kmh = 3.6 * config.rl.train_speed_min;
        const double span_kmh = 3.6 * (config.rl.train_speed_max -
                                       config.rl.train_speed_min);
        constexpr std::array<double, 11> selected_template{{
            30.0, 35.0, 36.0, 40.0, 45.0, 50.0,
            60.0, 70.0, 75.0, 90.0, 120.0}};
        for (const double speed : selected_template)
            selection_speeds_kmh.push_back(min_kmh +
                (speed - 30.0) / 90.0 * span_kmh);
        const auto source_info = readCarSimRunInfo(training_source);
        if (!source_info.speed_target_constant.empty())
            selection_speeds_kmh.push_back(
                std::stod(source_info.speed_target_constant));
    }
    for (const auto [speed_kmh, case_mu] : cases) {
        if (fixed_grip_evaluation) {
            if (speed_kmh < 3.6 * config.rl.train_speed_min - 1e-9 ||
                speed_kmh > 3.6 * config.rl.train_speed_max + 1e-9 ||
                std::any_of(selection_speeds_kmh.begin(),
                    selection_speeds_kmh.end(), [&](double selected_speed) {
                        return std::abs(speed_kmh - selected_speed) < 1e-5;
                    }) ||
                !evaluated_speeds.insert(speed_kmh).second)
                continue;
        }
        double mu = fixed_grip_evaluation ? config.rl.train_mu_min : case_mu;
        EpisodeResult basic;
        if (general_evaluation) {
            basic = runEpisode(basic_config, speed_kmh / 3.6, nullptr, false, mu);
            if (config.rl.preserve_source_road)
                mu = readCarSimMpcModel(config.output / "training_solver").road_grip.at(0.0);
            if (!same_source())
                throw std::runtime_error("CarSim source changed during general evaluation");
            if (peak_reference_curvature < 0.0) {
                const auto reference = DoubleLaneChange::fromCarSimParsfile(
                    config.output / "training_solver" / "Run_all.par");
                peak_reference_curvature = 0.0;
                double peak_x = 0.0;
                for (double x = 0.0; x <= config.distance_m; x += 0.1) {
                    const double curvature = std::abs(reference.at(x).curvature);
                    if (curvature > peak_reference_curvature) {
                        peak_reference_curvature = curvature;
                        peak_x = x;
                    }
                }
                std::ofstream manifest(config.output / "evaluation_manifest.txt", std::ios::app);
                manifest << std::setprecision(12)
                         << "reference_peak_abs_curvature_per_m=" << peak_reference_curvature << '\n'
                         << "reference_peak_curvature_x_m=" << peak_x << '\n'
                         << "demand_estimate=target_speed_squared_times_curvature_over_mu_g; ideal_flat_road_approximation\n";
            }
            const double speed_mps = speed_kmh / 3.6;
            const double ay = speed_mps * speed_mps * peak_reference_curvature;
            const double ideal_limit = peak_reference_curvature > 0.0 ?
                3.6 * std::sqrt(mu * 9.81 / peak_reference_curvature) : 0.0;
            demand_log << speed_kmh << ',' << mu << ',' << peak_reference_curvature << ','
                       << ay << ',' << mu * 9.81 << ',' << ay / (mu * 9.81) << ','
                       << ideal_limit << '\n';
        }
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
            if (general_evaluation) {
                grid_metrics[{mu, speed_kmh, name}] =
                    {result.rmse, result.max_abs_error, result.completed};
                auto& agg = aggregates[{mu, name}];
                ++agg.cases;
                agg.completed += result.completed ? 1 : 0;
                agg.rmse_sum += result.rmse;
                agg.worst_error = std::max(agg.worst_error, result.max_abs_error);
                agg.reward_sum += result.mean_reward;
            }
        };
        if (general_evaluation) write("fixed_mpc", basic);
        write(general_evaluation ? "rl_neutral" : "fixed", fixed);
        write(config.rl.algorithm.c_str(), learned);
        out.flush();
        std::cout << "Held-out " << speed_kmh << " km/h mu=" << mu
                  << " fixed RMSE/max=" << fixed.rmse << '/' << fixed.max_abs_error
                  << ' ' << config.rl.algorithm << " RMSE/max=" << learned.rmse << '/'
                  << learned.max_abs_error << '\n';
    }
    if (general_evaluation) {
        std::ofstream summary(config.output / "generalization_summary.csv", std::ios::binary);
        if (!summary) throw std::runtime_error("Cannot write generalization summary");
        summary << "mu,controller,cases,completed,mean_rmse_y_m,worst_max_abs_y_m,mean_step_reward\n"
                << std::setprecision(12);
        for (const auto& entry : aggregates) {
            const auto& agg = entry.second;
            summary << entry.first.first << ',' << entry.first.second << ','
                    << agg.cases << ',' << agg.completed << ','
                    << agg.rmse_sum / agg.cases << ',' << agg.worst_error << ','
                    << agg.reward_sum / agg.cases << '\n';
        }
        std::set<double> speeds, grips;
        for (const auto& item : grid_metrics) {
            grips.insert(std::get<0>(item.first));
            speeds.insert(std::get<1>(item.first));
        }
        std::ofstream html(config.output / "generalization_report.html", std::ios::binary);
        if (!html) throw std::runtime_error("Cannot write generalization report");
        html << "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                "<title>速度与附着泛化验证</title><style>"
                "body{font:15px 'Microsoft YaHei',sans-serif;max-width:1400px;margin:30px auto;padding:0 18px;"
                "background:#edf3f8;color:#183148}p{line-height:1.7}table{border-collapse:collapse;background:white;"
                "width:100%;margin:12px 0 30px}th,td{border:1px solid #cbd9e5;padding:8px;text-align:center}"
                "th{background:#e4eef8}td small{display:block;color:#52657c}.good{background:#e2f5e7}"
                ".worse{background:#fff0ed}.fail{background:#ffcfca}.limit{background:#ffedc5}"
                ".scroll{overflow-x:auto}a{color:#176eb8}</style>"
                "<h1>跨速度、附着系数验证</h1>"
             << (config.rl.preserve_source_road ?
                "<p>所有速度均使用原始道路附着表；μ 列为初始位置附着值，后续位置可能不同。理想侧向需求仅作参考。</p>" :
                "<p>μ 列为初始位置附着值；若源道路使用 STEP 表，则各工况按比例缩放整张表，后续位置的 μ 随道路变化。理想侧向需求表使用初始 μ，只是参考，不能代表全路段能力。</p>")
             <<
                "<p>每格显示横向 RMSE（m）、最大横向误差（m）及是否完成。"
                "绿色表示该策略的 RMSE 低于同工况基础 MPC，红色表示较高；"
                "颜色只表示相对变化，完成仿真和绿色均不代表绝对误差达标。</p>"
                "<p><a href=\"generalization_validation.csv\">逐工况数据</a> · "
                "<a href=\"generalization_summary.csv\">按 μ 汇总</a> · "
                "<a href=\"reference_demand.csv\">参考轨迹侧向需求</a> · "
                "<a href=\"evaluation_manifest.txt\">工况与来源</a> · "
                "<a href=\"policy_used.txt\">本次策略</a></p>";
        const auto grid_header = [&] {
            html << "<div class=\"scroll\"><table><thead><tr><th>μ / km/h</th>";
            for (double speed : speeds) html << "<th>" << std::fixed << std::setprecision(0) << speed << "</th>";
            html << "</tr></thead><tbody>";
        };
        for (const std::string controller : {std::string("fixed_mpc"), std::string("rl_neutral"), config.rl.algorithm}) {
            html << "<h2>" << (controller == "fixed_mpc" ? "基础 MPC" :
                controller == "rl_neutral" ? "RL 中性 MPC" :
                controller == "td3" ? "TD3-MPC" : "DDPG-MPC") << "</h2>";
            grid_header();
            for (double grip : grips) {
                html << "<tr><th>" << std::fixed << std::setprecision(2) << grip << "</th>";
                for (double speed : speeds) {
                    const auto found = grid_metrics.find({grip, speed, controller});
                    const auto baseline = grid_metrics.find({grip, speed, "fixed_mpc"});
                    if (found == grid_metrics.end() || baseline == grid_metrics.end()) {
                        html << "<td>—</td>";
                        continue;
                    }
                    const auto& value = found->second;
                    const auto& base = baseline->second;
                    const char* style = !value.completed ? "fail" :
                        controller == "fixed_mpc" ? "" :
                        value.rmse <= base.rmse ? "good" : "worse";
                    html << "<td class=\"" << style << "\">" << std::fixed << std::setprecision(3)
                         << value.rmse << "<small>最大 " << value.maximum
                         << (value.completed ? " · 完成" : " · 未完成") << "</small></td>";
                }
                html << "</tr>";
            }
            html << "</tbody></table></div>";
        }
        html << "<h2>参考轨迹的理想侧向需求</h2>"
                "<p>参考曲率最大值约为 " << std::setprecision(5) << peak_reference_curvature
             << " 1/m。下表按目标车速计算 v²|κ|max/(μg)；大于 1 表示理想平路近似下"
                "参考侧向加速度已超过 μg，黄色用于提示该速度与轨迹组合需求较高。"
                "实际能力还受轮胎、载荷和车辆动态影响。</p>";
        grid_header();
        for (double grip : grips) {
            html << "<tr><th>" << std::fixed << std::setprecision(2) << grip << "</th>";
            for (double speed : speeds) {
                const double v = speed / 3.6;
                const double ratio = v * v * peak_reference_curvature / (grip * 9.81);
                html << "<td class=\"" << (ratio > 1.0 ? "limit" : "") << "\">"
                     << std::fixed << std::setprecision(2) << ratio << "</td>";
            }
            html << "</tr>";
        }
        html << "</tbody></table></div></html>";
        if (!html) throw std::runtime_error("Failed writing generalization report");
    }
    std::ofstream config_file(config.output / "effective_config.ini", std::ios::binary);
    config_file << configToIni(config);
    std::cout << "Held-out results: "
              << std::filesystem::absolute(output_file).string()
              << '\n';
}
