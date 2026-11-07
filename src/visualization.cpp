#include "visualization.h"
#include "rl.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
constexpr double pi = 3.14159265358979323846264338327950288;
constexpr double rad2deg = 180.0 / pi;
using XY = std::pair<double, double>;
struct Trace {
    std::string name;
    std::string color;
    std::vector<XY> xy;
    bool dashed = false;
    double stroke_width = 3.1;
    std::string dash_pattern;
    double reference_min = std::numeric_limits<double>::quiet_NaN();
    double reference_max = std::numeric_limits<double>::quiet_NaN();
};

std::ofstream openFile(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create output file: " + path.string());
    return out;
}
std::string xmlEscape(const std::string& value) {
    std::string result;
    for (const char c : value) {
        switch (c) {
        case '&': result += "&amp;"; break;
        case '<': result += "&lt;"; break;
        case '>': result += "&gt;"; break;
        case '"': result += "&quot;"; break;
        default: result += c;
        }
    }
    return result;
}
double niceNumber(double value, bool round_result) {
    if (value <= 0 || !std::isfinite(value)) return 1.0;
    const double exponent = std::floor(std::log10(value));
    const double power = std::pow(10.0, exponent);
    const double fractional = value / power;
    double nice = 1.0;
    if (round_result) {
        if (fractional < 1.5) nice = 1.0;
        else if (fractional < 3.0) nice = 2.0;
        else if (fractional < 7.0) nice = 5.0;
        else nice = 10.0;
    } else {
        if (fractional <= 1.0) nice = 1.0;
        else if (fractional <= 2.0) nice = 2.0;
        else if (fractional <= 5.0) nice = 5.0;
        else nice = 10.0;
    }
    return nice * power;
}
std::string tickText(double value, double step) {
    int decimals = 0;
    if (step < 1.0) decimals = std::min(6, std::max(0,
                                      static_cast<int>(std::ceil(-std::log10(step))) + 1));
    if (std::abs(value) < step * 1.0e-6) value = 0.0;
    std::ostringstream os;
    os << std::fixed << std::setprecision(decimals) << value;
    return os.str();
}

void svgChart(const std::filesystem::path& file,
              const std::string& title,
              const std::string& xlabel,
              const std::string& ylabel,
              const std::vector<Trace>& traces,
              bool integer_y = false) {
    constexpr int width = 1160, height = 620;
    constexpr double left = 180, right = 40, bottom = 102;
    const double top = traces.size() > 4 ? 132.0 : 103.0;
    const double chart_width = width - left - right;
    const double chart_height = height - top - bottom;
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    for (const auto& tr : traces) {
        for (auto [x, y] : tr.xy) {
            if (std::isfinite(x) && std::isfinite(y)) {
                xmin = std::min(xmin, x);
                xmax = std::max(xmax, x);
                ymin = std::min(ymin, y);
                ymax = std::max(ymax, y);
            }
        }
    }
    if (!std::isfinite(xmin)) throw std::runtime_error("Plot has no finite data");
    if (xmax <= xmin) xmax = xmin + 1.0;
    if (ymax <= ymin) { ymax += 0.1; ymin -= 0.1; }
    double padding = 0.08 * (ymax - ymin);
    if (padding <= 0) padding = 0.1;
    ymin -= padding;
    ymax += padding;
    const double xstep = niceNumber((xmax - xmin) / 8.0, true);
    const double ystep = integer_y
        ? std::max(1.0, niceNumber((ymax - ymin) / 6.0, true))
        : niceNumber((ymax - ymin) / 6.0, true);
    const bool spatial = xlabel == "Longitudinal position X (m)";
    const bool has_reference_range = spatial && !traces.empty() &&
        std::isfinite(traces.front().reference_min) &&
        std::isfinite(traces.front().reference_max) &&
        traces.front().reference_max > traces.front().reference_min;
    const double xaxismin = has_reference_range ? traces.front().reference_min
                                                : std::floor(xmin / xstep) * xstep;
    const double xaxismax = has_reference_range ? traces.front().reference_max
                                                : std::ceil(xmax / xstep) * xstep;
    const double yaxismin = std::floor(ymin / ystep) * ystep;
    const double yaxismax = std::ceil(ymax / ystep) * ystep;
    const auto px = [&](double x) {
        return left + (x - xaxismin) / (xaxismax - xaxismin) * chart_width;
    };
    const auto py = [&](double y) {
        return top + (yaxismax - y) / (yaxismax - yaxismin) * chart_height;
    };

    auto out = openFile(file);
    out << std::fixed << std::setprecision(4);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << ' ' << height << "\">\n"
        << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n"
        << "<text x=\"" << (width / 2) << "\" y=\"44\" text-anchor=\"middle\" "
        << "font-family=\"Segoe UI,Arial,sans-serif\" font-size=\"26\" "
        << "font-weight=\"700\" fill=\"#152b45\">" << xmlEscape(title) << "</text>\n"
        << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << chart_width
        << "\" height=\"" << chart_height << "\" fill=\"#f9fbfe\" stroke=\"#cbd5e1\"/>\n";

    for (int k = 0; k <= 30; ++k) {
        const double x = xaxismin + k * xstep;
        if (x > xaxismax + xstep * 1.0e-6) break;
        out << "<line x1=\"" << px(x) << "\" x2=\"" << px(x)
            << "\" y1=\"" << top << "\" y2=\"" << top + chart_height
            << "\" stroke=\"#e4eaf1\" stroke-width=\"1\"/>\n"
            << "<text x=\"" << px(x) << "\" y=\"" << top + chart_height + 26
            << "\" text-anchor=\"middle\" font-family=\"Segoe UI,Arial\" font-size=\"15\" "
            << "fill=\"#52657c\">" << tickText(x, xstep) << "</text>\n";
    }
    if (has_reference_range &&
        std::abs((xaxismax - xaxismin) / xstep -
                 std::round((xaxismax - xaxismin) / xstep)) > 1e-6) {
        out << "<text x=\"" << px(xaxismax) << "\" y=\"" << top + chart_height + 26
            << "\" text-anchor=\"middle\" font-family=\"Segoe UI,Arial\" font-size=\"15\" "
            << "fill=\"#52657c\">" << tickText(xaxismax, xstep) << "</text>\n";
    }
    for (int k = 0; k <= 30; ++k) {
        const double y = yaxismin + k * ystep;
        if (y > yaxismax + ystep * 1.0e-6) break;
        out << "<line x1=\"" << left << "\" x2=\"" << left + chart_width
            << "\" y1=\"" << py(y) << "\" y2=\"" << py(y)
            << "\" stroke=\"#e4eaf1\" stroke-width=\"1\"/>\n"
            << "<text x=\"" << left - 13 << "\" y=\"" << py(y) + 5
            << "\" text-anchor=\"end\" font-family=\"Segoe UI,Arial\" font-size=\"15\" "
            << "fill=\"#52657c\">" << tickText(y, ystep) << "</text>\n";
    }
    out << "<text x=\"" << left + chart_width / 2
        << "\" y=\"" << height - 25 << "\" text-anchor=\"middle\" "
        << "font-family=\"Segoe UI,Arial\" font-size=\"19\" fill=\"#33475d\">"
        << xmlEscape(xlabel) << "</text>\n"
        << "<text transform=\"translate(36," << top + chart_height / 2
        << ") rotate(-90)\" text-anchor=\"middle\" "
        << "font-family=\"Segoe UI,Arial\" font-size=\"19\" fill=\"#33475d\">"
        << xmlEscape(ylabel) << "</text>\n";
    out << "<defs><clipPath id=\"plotclip\"><rect x=\"" << left << "\" y=\"" << top
        << "\" width=\"" << chart_width << "\" height=\"" << chart_height
        << "\"/></clipPath></defs>\n";
    for (const auto& tr : traces) {
        out << "<polyline fill=\"none\" clip-path=\"url(#plotclip)\" "
            << "stroke=\"" << tr.color << "\" stroke-width=\"" << tr.stroke_width << "\" "
            << "stroke-linecap=\"round\" stroke-linejoin=\"round\" ";
        if (tr.dashed || !tr.dash_pattern.empty())
            out << "stroke-dasharray=\"" << (tr.dash_pattern.empty() ? "11,7" : tr.dash_pattern) << "\" ";
        out << "points=\"";
        for (const auto& [x, y] : tr.xy)
            if (std::isfinite(x) && std::isfinite(y)) out << px(x) << ',' << py(y) << ' ';
        out << "\"/>\n";
    }
    // A second legend row keeps five or more traces within the SVG width.
    const std::size_t rows = traces.size() > 4 ? 2 : 1;
    const std::size_t per_row = (traces.size() + rows - 1) / rows;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t begin = row * per_row;
        const std::size_t end = std::min(traces.size(), begin + per_row);
        double xlegend = width - 40;
        const int legend_y = rows == 1 ? 76 : 73 + static_cast<int>(row) * 28;
        for (std::size_t k = end; k-- > begin;) {
            const auto& tr = traces[k];
            const double label_width = 13.0 * tr.name.size() + 56.0;
            xlegend -= label_width;
            out << "<line x1=\"" << xlegend << "\" y1=\"" << legend_y
                << "\" x2=\"" << xlegend + 32 << "\" y2=\"" << legend_y
                << "\" stroke-width=\"" << tr.stroke_width << "\" stroke=\""
                << tr.color << "\"";
            if (tr.dashed || !tr.dash_pattern.empty())
                out << " stroke-dasharray=\""
                    << (tr.dash_pattern.empty() ? "11,7" : tr.dash_pattern) << "\"";
            out << "/>\n<text x=\"" << xlegend + 39
                << "\" y=\"" << legend_y + 6
                << "\" font-family=\"Segoe UI,Arial\" font-size=\"16\" "
                << "fill=\"#33475d\">" << xmlEscape(tr.name) << "</text>\n";
        }
    }
    out << "</svg>\n";
}

template <typename XFn, typename YFn>
Trace makeTrace(const std::vector<SimulationRecord>& samples,
                const std::string& name, const std::string& color,
                XFn getx, YFn gety, bool dashed = false) {
    Trace out;
    out.name = name;
    out.color = color;
    out.dashed = dashed;
    out.xy.reserve(samples.size());
    if (!samples.empty()) {
        out.reference_min = samples.front().reference.x_min;
        out.reference_max = samples.front().reference.x_max;
    }
    const bool has_range = std::isfinite(out.reference_min) &&
        std::isfinite(out.reference_max) && out.reference_max > out.reference_min;
    for (const auto& row : samples) {
        if (!has_range || (row.s.x >= out.reference_min && row.s.x <= out.reference_max))
            out.xy.emplace_back(getx(row), gety(row));
    }
    return out;
}
std::string fmt(double value, int precision = 4) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}
} // namespace

SimulationRecord recordSample(const VehicleState& state, const DoubleLaneChange& reference,
                              double time, const MpcResult& command, double solve_ms,
                              bool command_sample, bool fallback_control,
                              const MpcTuningAction& action, double road_grip,
                              double curvature_sigma_m) {
    SimulationRecord row;
    row.time = time;
    row.s = state;
    row.reference = reference.at(state.x);
    row.lateral_error = state.y - row.reference.y;
    row.heading_error = angleDifference(state.psi, row.reference.psi);
    row.steer = command.steer;
    row.yaw_rate_ref = state.vx * row.reference.curvature;
    row.feedforward_curvature = reference.controlCurvature(state.x, curvature_sigma_m);
    row.road_grip = road_grip;
    row.qp_iterations = command.iterations;
    row.qp_primal_residual = command.primal_residual;
    row.qp_dual_residual = command.dual_residual;
    row.qp_solve_ms = solve_ms;
    row.qp_converged = command.converged;
    row.is_command = command_sample;
    row.fallback_control = fallback_control;
    row.prediction_horizon = action.np;
    row.q_y = action.q_y;
    row.q_psi = action.q_psi;
    row.q_vy = action.q_vy;
    row.q_r = action.q_r;
    row.r_steer = action.r_steer;
    row.r_delta = action.r_delta;
    return row;
}

SimulationMetrics exportResults(
    const std::filesystem::path& output,
    const std::vector<SimulationRecord>& rows,
    const VehicleParameters& vp,
    const MpcParameters& mp,
    const std::string& experiment_name,
    const RunOutcome& outcome) {
    if (rows.empty()) throw std::invalid_argument("Simulation yielded no samples");
    std::filesystem::create_directories(output);
    SimulationMetrics metrics;
    metrics.min_speed_mps = rows.front().s.vx;
    metrics.max_speed_mps = rows.front().s.vx;
    metrics.min_ground_speed_mps = std::hypot(rows.front().s.vx, rows.front().s.vy);
    metrics.max_ground_speed_mps = metrics.min_ground_speed_mps;
    double sum_abs_y = 0.0, sum_y2 = 0.0, sum_heading2 = 0.0, sum_iters = 0.0;
    double sum_abs_yaw_rate = 0.0, sum_abs_front_wheel = 0.0;
    std::size_t front_wheel_samples = 0;
    double sum_solve_ms = 0.0;
    int solver_samples = 0;
    int min_np = rows.front().prediction_horizon;
    int max_np = min_np;
    const auto applied_weights = [](const SimulationRecord& p) {
        return std::array<double, 6>{p.q_y, p.q_psi,
            p.q_vy, p.q_r, p.r_steer, p.r_delta};
    };
    auto min_weights = applied_weights(rows.front());
    auto max_weights = min_weights;
    double prev_delta = 0.0;
    for (const auto& p : rows) {
        min_np = std::min(min_np, p.prediction_horizon);
        max_np = std::max(max_np, p.prediction_horizon);
        const auto current_weights = applied_weights(p);
        for (std::size_t k = 0; k < 6; ++k) {
            min_weights[k] = std::min(min_weights[k], current_weights[k]);
            max_weights[k] = std::max(max_weights[k], current_weights[k]);
        }
        metrics.min_speed_mps = std::min(metrics.min_speed_mps, p.s.vx);
        metrics.max_speed_mps = std::max(metrics.max_speed_mps, p.s.vx);
        const double ground_speed = std::hypot(p.s.vx, p.s.vy);
        metrics.min_ground_speed_mps = std::min(metrics.min_ground_speed_mps, ground_speed);
        metrics.max_ground_speed_mps = std::max(metrics.max_ground_speed_mps, ground_speed);
        sum_abs_y += std::abs(p.lateral_error);
        sum_y2 += p.lateral_error * p.lateral_error;
        sum_heading2 += p.heading_error * p.heading_error;
        metrics.max_abs_y = std::max(metrics.max_abs_y, std::abs(p.lateral_error));
        metrics.max_abs_heading_deg = std::max(metrics.max_abs_heading_deg,
                                               std::abs(p.heading_error * rad2deg));
        const double abs_yaw_rate = std::abs(p.s.r) * rad2deg;
        sum_abs_yaw_rate += abs_yaw_rate;
        metrics.max_abs_yaw_rate_deg_s = std::max(metrics.max_abs_yaw_rate_deg_s, abs_yaw_rate);
        if (std::isfinite(p.s.steer_front_left) && std::isfinite(p.s.steer_front_right)) {
            const double left = std::abs(p.s.steer_front_left) * rad2deg;
            const double right = std::abs(p.s.steer_front_right) * rad2deg;
            sum_abs_front_wheel += left + right;
            metrics.max_abs_front_wheel_deg = std::max({metrics.max_abs_front_wheel_deg, left, right});
            ++front_wheel_samples;
        }
        metrics.max_abs_steering_deg = std::max(metrics.max_abs_steering_deg,
                                                std::abs(p.steer * rad2deg));
        metrics.max_steering_rate_deg_s = std::max(metrics.max_steering_rate_deg_s,
                                         std::abs((p.steer - prev_delta) / mp.dt * rad2deg));
        if (p.is_command) {
            ++solver_samples;
            sum_iters += p.qp_iterations;
            sum_solve_ms += p.qp_solve_ms;
            metrics.max_qp_time_ms = std::max(metrics.max_qp_time_ms, p.qp_solve_ms);
            metrics.qp_nonconverged_steps += !p.qp_converged ? 1 : 0;
            metrics.fallback_control_steps += p.fallback_control ? 1 : 0;
            metrics.peak_qp_primal_residual = std::max(metrics.peak_qp_primal_residual,
                                                      p.qp_primal_residual);
            metrics.peak_qp_dual_residual = std::max(metrics.peak_qp_dual_residual,
                                                    p.qp_dual_residual);
        }
        prev_delta = p.steer;
    }
    metrics.mae_y = sum_abs_y / rows.size();
    metrics.rmse_y = std::sqrt(sum_y2 / rows.size());
    metrics.rmse_heading_deg = rad2deg * std::sqrt(sum_heading2 / rows.size());
    metrics.mean_abs_yaw_rate_deg_s = sum_abs_yaw_rate / rows.size();
    metrics.has_front_wheel_angles = front_wheel_samples == rows.size();
    if (metrics.has_front_wheel_angles)
        metrics.mean_abs_front_wheel_deg = sum_abs_front_wheel / (2.0 * front_wheel_samples);
    if (solver_samples > 0) {
        metrics.mean_qp_iterations = sum_iters / solver_samples;
        metrics.mean_qp_time_ms = sum_solve_ms / solver_samples;
    }

    {
        auto out = openFile(output / "simulation.csv");
        out << "time_s,vx_carsim_kmh,ground_speed_carsim_kmh,X_carsim_m,Y_carsim_m,"
                   "Y_ref_from_carsim_table_m,ey_m,psi_carsim_deg,psi_ref_deg,epsi_deg,"
                   "front_steer_command_deg,vy_carsim_kmh,yaw_rate_carsim_degps,"
                   "yaw_rate_ref_degps,qp_iterations,qp_primal_residual,qp_dual_residual,"
                   "qp_solve_ms,qp_converged,is_command,fallback_control,np,"
                   "q_y_applied,q_psi_applied,q_vy_applied,"
                   "q_r_applied,r_steer_applied,r_delta_applied,road_grip_estimate,reward,"
                   "reference_curvature_inv_m,front_lagged_slip_rad,rear_lagged_slip_rad,"
                   "front_left_wheel_steer_deg,front_right_wheel_steer_deg,feedforward_curvature_inv_m\n";
        out << std::fixed << std::setprecision(9);
        for (const auto& p : rows) {
                out << p.time << ',' << 3.6 * p.s.vx << ',' << 3.6 * std::hypot(p.s.vx, p.s.vy)
                    << ',' << p.s.x << ',' << p.s.y << ',' << p.reference.y << ','
                    << p.lateral_error << ',' << p.s.psi * rad2deg << ','
                    << p.reference.psi * rad2deg << ',' << p.heading_error * rad2deg
                    << ',' << p.steer * rad2deg << ',' << 3.6 * p.s.vy << ','
                    << p.s.r * rad2deg << ',' << p.yaw_rate_ref * rad2deg
                    << ',' << p.qp_iterations << ',' << p.qp_primal_residual << ','
                    << p.qp_dual_residual << ',' << p.qp_solve_ms << ','
                    << (p.qp_converged ? 1 : 0) << ',' << (p.is_command ? 1 : 0)
                    << ',' << (p.fallback_control ? 1 : 0) << ','
                    << p.prediction_horizon;
                out << ',' << p.q_y << ',' << p.q_psi << ','
                    << p.q_vy << ',' << p.q_r << ','
                    << p.r_steer << ',' << p.r_delta << ',' << p.road_grip << ',' << p.reward
                    << ',' << p.reference.curvature << ',' << p.s.alpha_front
                    << ',' << p.s.alpha_rear << ',';
                if (std::isfinite(p.s.steer_front_left)) out << p.s.steer_front_left * rad2deg;
                out << ',';
                if (std::isfinite(p.s.steer_front_right)) out << p.s.steer_front_right * rad2deg;
                out << ',' << p.feedforward_curvature << '\n';
        }
    }

    auto xcoord = [](const SimulationRecord& v) { return v.s.x; };
    auto constant0 = [](const SimulationRecord&) { return 0.0; };
    const std::string blue = "#147bd1", orange = "#e96932", red = "#c83e58";
    svgChart(output / "01_lateral_tracking.svg", "CarSim reference: lateral tracking",
             "Longitudinal position X (m)", "Lateral position Y (m)", {
             makeTrace(rows, "MPC actual", blue, xcoord,
                       [](const auto& v) {return v.s.y;}),
             makeTrace(rows, "Reference", orange, xcoord,
                       [](const auto& v) {return v.reference.y;}, true)});
    svgChart(output / "02_sideslip.svg", "Vehicle side-slip angle",
             "Longitudinal position X (m)", "Side-slip angle (deg)", {
             makeTrace(rows, "CarSim actual", blue, xcoord,
                       [](const auto& v) {return std::atan2(v.s.vy, v.s.vx) * rad2deg;})});
    svgChart(output / "02_lateral_velocity.svg", "Vehicle lateral velocity",
             "Longitudinal position X (m)", "Lateral velocity (m/s)", {
             makeTrace(rows, "CarSim actual", blue, xcoord,
                       [](const auto& v) {return v.s.vy;})});
    svgChart(output / "02_heading.svg", "Vehicle heading angle",
             "Longitudinal position X (m)", "Heading angle (deg)", {
             makeTrace(rows, "CarSim actual", blue, xcoord,
                       [](const auto& v) {return v.s.psi * rad2deg;})});
    svgChart(output / "02_yaw_rate.svg", "Vehicle yaw rate",
             "Longitudinal position X (m)", "Yaw rate (deg/s)", {
             makeTrace(rows, "CarSim actual", blue, xcoord,
                       [](const auto& v) {return v.s.r * rad2deg;})});
    svgChart(output / "03_lateral_error.svg", "Lateral tracking error",
             "Longitudinal position X (m)", "Y - Yref (m)", {
             makeTrace(rows, "Zero", "#a4b0bf", xcoord, constant0, true),
             makeTrace(rows, "Lateral error", red, xcoord,
                       [](const auto& v) {return v.lateral_error;})});
    svgChart(output / "04_heading_error.svg", "Heading tracking error",
             "Longitudinal position X (m)", "Psi - Psiref (deg)", {
             makeTrace(rows, "Zero", "#a4b0bf", xcoord, constant0, true),
             makeTrace(rows, "Heading error", red, xcoord,
                       [](const auto& v) {return v.heading_error * rad2deg;})});
    const bool filtered_curvature = mp.steer_feedforward && mp.r_steer > 0.0 &&
                                    mp.reference_curvature_sigma_m > 0.0;
    svgChart(output / "06_reference_curvature.svg",
             filtered_curvature ? "MPC steering feedforward curvature" : "CarSim reference curvature",
             "Longitudinal position X (m)", "Curvature (1/m)", {
             makeTrace(rows, filtered_curvature ? "Spatial estimate" : "Reference", orange, xcoord,
                       [filtered_curvature](const auto& v) {
                           return filtered_curvature ? v.feedforward_curvature : v.reference.curvature;
                       })});
    if (filtered_curvature)
        svgChart(output / "06_native_reference_curvature.svg", "CarSim native reference curvature",
                 "Longitudinal position X (m)", "Curvature (1/m)", {
                 makeTrace(rows, "Native geometry", orange, xcoord,
                           [](const auto& v) {return v.reference.curvature;})});
    std::vector<Trace> steering_traces{
        makeTrace(rows, "MPC command", blue, xcoord, [](const auto& v) {return v.steer * rad2deg;})};
    if (metrics.has_front_wheel_angles) {
        steering_traces.push_back(makeTrace(rows, "CarSim front left", orange, xcoord,
            [](const auto& v) {return v.s.steer_front_left * rad2deg;}));
        steering_traces.push_back(makeTrace(rows, "CarSim front right", red, xcoord,
            [](const auto& v) {return v.s.steer_front_right * rad2deg;}));
    }
    svgChart(output / "05_steering.svg", "Front-wheel steering",
             "Longitudinal position X (m)", "Front steering angle (deg)", steering_traces);
    svgChart(output / "08_prediction_horizon.svg", "Prediction horizon selected by controller",
             "Longitudinal position X (m)", "Np (steps)", {
             makeTrace(rows, "Applied Np", blue, xcoord,
                       [](const auto& v) {return static_cast<double>(v.prediction_horizon);})}, true);
    svgChart(output / "09_q_y_value.svg", "Applied MPC Qy",
             "Longitudinal position X (m)", "Qy absolute value", {
             makeTrace(rows, "Qy", blue, xcoord, [](const auto& v){return v.q_y;})});
    svgChart(output / "09_q_psi_value.svg", "Applied MPC Qpsi",
             "Longitudinal position X (m)", "Qpsi absolute value", {
             makeTrace(rows, "Qpsi", blue, xcoord, [](const auto& v){return v.q_psi;})});
    svgChart(output / "10_r_delta_value.svg", "Applied MPC Rdelta",
             "Longitudinal position X (m)", "Rdelta absolute value", {
             makeTrace(rows, "Rdelta", blue, xcoord, [](const auto& v){return v.r_delta;})});

    {
        auto out = openFile(output / "summary.txt");
        out << "CarSim VS Solver"
            << " / Lateral MPC | Double-Lane-Change Simulation\n"
            << "========================================================\n"
            << "C++ experiment label: " << experiment_name << '\n'
            << "Reference source: " << outcome.reference_source << '\n'
            << (outcome.adaptive ? "Controller mode: Python RL adaptive Np and Q/R\n"
                                 : "Controller mode: fixed prediction horizon and Q/R weights\n")
            << "Run status: " << (outcome.completed ? "completed" : "incomplete") << '\n'
            << "Requested distance (m): " << fmt(outcome.requested_distance_m, 3) << '\n'
            << "Stop reason: " << (outcome.completed ? "none" : outcome.stop_reason) << '\n'
            << "First MPC solve issue: "
            << (outcome.first_solve_issue.empty() ? "none" : outcome.first_solve_issue)
            << '\n';
        {
            const auto& source = outcome.carsim_info;
            out << "CarSim source: carsim_native/Run_all.par (snapshot copied at launch)\n";
            out << "CarSim Browser Video/Plot cache: " << outcome.browser_result_status << '\n';
            if (!outcome.browser_result_file.empty())
                out << "CarSim Browser result header: "
                    << outcome.browser_result_file.string() << '\n';
            if (std::filesystem::exists(output / "carsim_native" / "open_visualizer.cmd"))
                out << "CarSim Visualizer launcher: carsim_native/open_visualizer.cmd\n";
            out << "CarSim Visualizer entry: carsim_native/visualizer.par\n"
                   "CarSim Visualizer source: carsim_native/Run_visualizer.par\n"
                   "CarSim native result header: carsim_native/LastRun.vs\n"
                   "CarSim native result data: carsim_native/LastRun.vsb\n";
            if (!source.vehicle_dataset.empty()) out << "CarSim vehicle dataset: " << source.vehicle_dataset << '\n';
            if (!source.road_dataset.empty()) out << "CarSim road dataset title: " << source.road_dataset << '\n';
            if (!source.procedure_dataset.empty()) out << "CarSim procedure dataset: " << source.procedure_dataset << '\n';
            if (!source.reference_dataset.empty()) out << "CarSim reference dataset: " << source.reference_dataset << '\n';
            if (!source.speed_target_constant.empty())
                out << "CarSim SPEED_TARGET_CONSTANT (km/h; solver echo): "
                    << source.speed_target_constant << '\n';
            if (!source.stop_time.empty()) out << "CarSim TSTOP (source value): " << source.stop_time << '\n';
            out << "CarSim observed initial Vx (km/h): " << fmt(3.6 * rows.front().s.vx, 4) << '\n';
        }
        out << "Body longitudinal vx range (km/h): " << fmt(3.6 * metrics.min_speed_mps, 4)
            << " / " << fmt(3.6 * metrics.max_speed_mps, 4) << '\n'
            << "Ground speed range (km/h): " << fmt(3.6 * metrics.min_ground_speed_mps, 4)
            << " / " << fmt(3.6 * metrics.max_ground_speed_mps, 4) << '\n'
            << "Sampling period (s): " << fmt(mp.dt, 4) << '\n'
            << "Prediction horizon Np (steps): " << mp.np << '\n'
            << "Applied Np range (steps): " << min_np << " / " << max_np << '\n'
            << "Control horizon Nc (steps): " << mp.nc << '\n'
            << "Prediction time Tp (s): " << fmt(mp.np * mp.dt, 4) << '\n'
            << "Control time Tc (s): " << fmt(mp.nc * mp.dt, 4) << '\n'
            << "Control blocking: hold last steering after Nc incremental moves\n";
        const char* weight_names[] = {"Q_y", "Q_psi", "Q_vy", "Q_r",
                                      "R_steer", "R_delta"};
        for (std::size_t k = 0; k < 6; ++k)
            out << "Applied " << weight_names[k] << " absolute range: "
                << fmt(min_weights[k], 6) << " / " << fmt(max_weights[k], 6) << '\n';
        out << "MPC prediction states: [vy,vx,psi,r,Y,X"
            << (mp.usesTireRelaxation() ? ",lagged front/rear tire slip]\n" : "]\n");
        if (mp.usesTireRelaxation())
            out << "Effective front / rear relaxation lengths at zero slip (m): "
                << fmt(mp.front_relaxation_m) << " / " << fmt(mp.rear_relaxation_m)
                << "\nCarSim lagged slip and lateral forces anchor the near-term tire prediction\n";
        out
                << "MPC augmentation: previous steering state, optimized steering increments\n"
                << "MPC discretization: stage-varying CarSim operating points, bilinear Tustin\n"
                << "MPC road grip preview: CarSim road map sampled at predicted X for each stage\n"
                << "Longitudinal input: measured consecutive CarSim body-vx rate; longitudinal tire slip/stiffness unavailable\n"
                << "MPC model source: this run's carsim_native/LastRun_echo.par\n"
                << "MPC total-laden mass (kg): " << fmt(vp.mass, 6) << '\n'
                << "MPC total-laden yaw inertia (kg m^2): " << fmt(vp.iz, 6) << '\n'
                << "MPC CG-to-front / CG-to-rear (m): " << fmt(vp.a, 6)
                << " / " << fmt(vp.b, 6) << '\n'
                << "Road grip at initial CG station: " << fmt(rows.front().road_grip, 6)
                << (outcome.variable_road_grip ? " (variable road map)\n" : " (constant road)\n")
                << "Initial MPC front / rear linearized axle stiffness (N/rad): "
                << fmt(outcome.model_front_stiffness_n_per_rad, 3) << " / "
                << fmt(outcome.model_rear_stiffness_n_per_rad, 3) << '\n'
                << "MPC tire source: carsim_native/carsim_tire_fy.csv (same four tire carpets)\n"
                << "MPC approximation: static axle loads; nonlinear tire table at each operating point with a slope floor; CarSim plant remains nonlinear\n";
        out
            << "Configured base Q diag: " << fmt(mp.q_y) << ", " << fmt(mp.q_psi)
            << ", " << fmt(mp.q_vy) << ", " << fmt(mp.q_r) << '\n'
            << "Configured base R steering feedforward deviation via increments: " << fmt(mp.r_steer) << '\n'
            << "Configured base R_delta increment penalty: " << fmt(mp.r_delta) << '\n'
            << "Configured R_delta_change penalty: " << fmt(mp.r_delta_change) << '\n'
            << "Steering feedforward curvature spatial sigma (m): "
            << fmt(mp.reference_curvature_sigma_m) << '\n'
            << "Terminal multiplier: " << fmt(mp.terminal_scale) << '\n'
            << "Prediction-step weight lambda: " << fmt(mp.stage_weight_lambda) << '\n'
            << "Steering limit (deg): " << fmt(mp.max_steer * rad2deg) << '\n'
            << "Steering limit source: "
            << (outcome.steer_limit_from_carsim ? "CarSim front RACK_KIN table range" : "explicit configuration") << '\n'
            << "Steering input: "
            << (outcome.steering_wheel_input ? "handwheel via CarSim rack map" : "configured direct input") << '\n'
            << "Steering slew limit (deg/s): "
            << (std::isfinite(mp.max_steer_rate) ? fmt(mp.max_steer_rate * rad2deg) : "none") << '\n'
            << "Speed-based slew schedule: " << (mp.speed_rate_enabled ? "enabled" : "disabled")
            << "; start/end (km/h): " << fmt(mp.speed_rate_start * 3.6) << " / " << fmt(mp.speed_rate_end * 3.6)
            << "; end limit (deg/s): " << fmt(mp.speed_rate_end_limit * rad2deg) << '\n'
            << "Simulation samples: " << rows.size() << '\n'
            << "Distance reached (m): " << fmt(rows.back().s.x, 3) << '\n'
            << "Reference parameters: CarSim LTARG_TABLE in staged Run_all.par\n"
            << "MPC angle objective: " << (mp.track_course ? "CG course angle" : "body yaw angle") << '\n'
            << "MAE lateral (m): " << fmt(metrics.mae_y, 6) << '\n'
            << "RMSE lateral (m): " << fmt(metrics.rmse_y, 6) << '\n'
            << "Max absolute lateral error (m): " << fmt(metrics.max_abs_y, 6) << '\n'
            << "RMSE heading (deg): " << fmt(metrics.rmse_heading_deg, 6) << '\n'
            << "Max absolute heading error (deg): " << fmt(metrics.max_abs_heading_deg, 6) << '\n'
            << "Max absolute yaw rate (deg/s): " << fmt(metrics.max_abs_yaw_rate_deg_s, 6) << '\n'
            << "Mean absolute yaw rate (deg/s): " << fmt(metrics.mean_abs_yaw_rate_deg_s, 6) << '\n'
            << "Max absolute actual front wheel angle (deg): "
            << (metrics.has_front_wheel_angles ? fmt(metrics.max_abs_front_wheel_deg, 6) : "unavailable") << '\n'
            << "Mean absolute actual front wheel angle (deg): "
            << (metrics.has_front_wheel_angles ? fmt(metrics.mean_abs_front_wheel_deg, 6) : "unavailable") << '\n'
            << "Max absolute front steering command (deg): "
            << fmt(metrics.max_abs_steering_deg, 6) << '\n'
            << "Max front steering command slew (deg/s): "
            << fmt(metrics.max_steering_rate_deg_s, 6) << '\n';
        out
            << "Average QP iterations per step: " << fmt(metrics.mean_qp_iterations, 2) << '\n'
            << "Mean QP solve time (ms): " << fmt(metrics.mean_qp_time_ms, 4) << '\n'
            << "Peak QP solve time (ms): " << fmt(metrics.max_qp_time_ms, 4) << '\n'
            << "Nonconverged QP steps: " << metrics.qp_nonconverged_steps << '\n'
            << "Note: includes prediction errors and QP iteration limits.\n"
            << "Held steering steps: " << metrics.fallback_control_steps << '\n';
        out
            << "Peak QP primal residual: " << fmt(metrics.peak_qp_primal_residual, 8) << '\n'
            << "Peak QP dual residual: " << fmt(metrics.peak_qp_dual_residual, 8) << '\n';
    }

    {
        auto out = openFile(output / "report.html");
        out << "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            << "<title>双移线 MPC 横向控制仿真报告</title>"
            << "<style>body{margin:0;background:#edf3f8;color:#183148;font-family:'Microsoft YaHei','Segoe UI',sans-serif}"
            << "header{background:#12375d;color:white;padding:34px max(20px,calc((100vw - 1240px)/2));}"
            << "header h1{font-size:30px;margin:0 0 12px}header p{opacity:.88;margin:0}"
            << "main{max-width:1240px;margin:25px auto;padding:0 20px}"
            << ".stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:15px;margin-bottom:28px}"
            << ".stat{background:white;border-radius:10px;padding:19px 21px;box-shadow:0 2px 9px #11253612;}"
            << ".stat small{color:#59718c}.stat strong{display:block;margin:12px 0 2px;color:#147bd1;font-size:26px}"
            << ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(510px,1fr));gap:18px}"
            << ".plot{background:white;padding:10px;border-radius:12px;box-shadow:0 2px 9px #11253612}"
            << ".plot img{display:block;width:100%;height:auto}"
            << "footer{max-width:1240px;margin:30px auto;padding:0 20px 30px;color:#59718c;line-height:1.8}"
            << ".notice{background:#fff3dd;border-left:5px solid #d99000;padding:14px 18px;"
               "border-radius:8px;margin-bottom:20px;line-height:1.7}"
            << "code{background:#dce8f2;padding:3px 5px;border-radius:4px}"
            << "@media(max-width:550px){.grid{grid-template-columns:1fr}header h1{font-size:22px}}"
            << "</style></head><body><header><h1>"
            << "CarSim 车辆动力学"
            << (outcome.adaptive ? " · Python 强化学习自适应 MPC 双移线轨迹跟踪</h1>"
                                 : " · 中性 MPC 双移线轨迹跟踪</h1>")
            << "<p>C++ 实验标签：" << xmlEscape(experiment_name)
            << " · 参考轨迹：" << xmlEscape(outcome.reference_source)
            << " · CarSim VS Solver / C++17 · MPC 闭环仿真</p>"
            << "</header><main>";
        {
            const auto& source = outcome.carsim_info;
            out << "<div class=\"notice\"><strong>本次 CarSim 工况快照</strong>"
                << "（读取启动时生成的 Run_all.par；CarSim 界面未重新生成的修改不会进入本次仿真）<br>";
            if (!source.vehicle_dataset.empty())
                out << "车辆数据集：" << xmlEscape(source.vehicle_dataset) << "<br>";
            if (!source.procedure_dataset.empty())
                out << "工况数据集：" << xmlEscape(source.procedure_dataset) << "<br>";
            if (!source.reference_dataset.empty())
                out << "轨迹数据集：" << xmlEscape(source.reference_dataset) << "<br>";
            if (!source.road_dataset.empty())
                out << "道路数据集标题：" << xmlEscape(source.road_dataset) << "<br>";
            out << "MPC 整车质量：" << fmt(vp.mass, 3) << " kg；偏航惯量："
                << fmt(vp.iz, 3) << " kg·m²<br>"
                << "质心到前/后轴：" << fmt(vp.a, 4) << " / " << fmt(vp.b, 4)
                << " m；初始路面摩擦系数：" << fmt(rows.front().road_grip, 3)
                << (outcome.variable_road_grip ? "（随道路表变化）" : "（常附着）") << "<br>"
                << "初始前/后轴小侧偏参考刚度："
                << fmt(outcome.model_front_stiffness_n_per_rad, 1) << " / "
                << fmt(outcome.model_rear_stiffness_n_per_rad, 1)
                << " N/rad（按该次轮胎表和静态轴荷计算）<br>";
            if (!source.speed_target_constant.empty())
                out << "CarSim SPEED_TARGET_CONSTANT 原值："
                    << xmlEscape(source.speed_target_constant)
                    << " km/h（经本次 CarSim Solver 回显确认；逐周期车速见 simulation.csv）<br>";
            if (!source.stop_time.empty())
                out << "CarSim TSTOP 原值：" << xmlEscape(source.stop_time) << "<br>";
            out << "<a href=\"carsim_native/Run_all.par\">查看本次 CarSim 参数快照</a></div>"
                << "<div class=\"notice\"><strong>CarSim 原生可视化</strong><br>"
                << "本次闭环另存了 <a href=\"carsim_native/LastRun.vs\">LastRun.vs</a>、"
                << "<a href=\"carsim_native/LastRun.vsb\">LastRun.vsb</a> 和"
                << " <a href=\"carsim_native/Run_visualizer.par\">完整可视化参数文件</a>。";
            if (std::filesystem::exists(output / "carsim_native" / "open_visualizer.cmd"))
                out << "在资源管理器中双击 <a href=\"carsim_native/open_visualizer.cmd\">open_visualizer.cmd</a>"
                    << " 可用 VS Visualizer 加载本次结果；";
            out << "也可在 VS Visualizer 中打开 <a href=\"carsim_native/visualizer.par\">visualizer.par</a>。"
                << "保留同目录的 LastRun.vsb。<br>";
            if (outcome.browser_result_status == "synchronized")
                out << "本次结果已同步到 CarSim Browser 的 LastRun 缓存；在对应 Run Control 点击 Refresh 后可用 Video/Plot。";
            else
                out << "CarSim Browser 缓存状态：" << xmlEscape(outcome.browser_result_status)
                    << "；独立结果仍保存在本目录。";
            out << "</div>";
        }
        if (!outcome.completed) {
            out << "<div class=\"notice\"><strong>仿真未完成：以下仅为已计算区间的真实结果。</strong><br>"
                << "已到达 X=" << fmt(rows.back().s.x, 2) << " m / 目标 "
                << fmt(outcome.requested_distance_m, 2) << " m；停止原因："
                << xmlEscape(outcome.stop_reason) << "</div>";
        } else if (metrics.qp_nonconverged_steps > 0) {
            out << "<div class=\"notice\">部分周期未得到收敛解。有限的迭代结果按转角约束投影，"
                << "计算异常时沿用上一周期转角。</div>";
        }
        out << "<div class=\"notice\"><strong>本次 MPC 模型与约束</strong><br>"
            << "控制器：增量式三自由度 LTV-MPC；轮胎预测模型："
            << "非线性轮胎表局部线性化"
            << "；转向输入：" << (outcome.steering_wheel_input ? "方向盘角" : "配置的直接输入")
            << "；前轮转角上限：" << fmt(mp.max_steer * rad2deg, 2) << " °"
            << (outcome.steer_limit_from_carsim ? "（CarSim 齿条运动学表范围）" : "（显式配置）")
            << "；转角速率上限："
            << (std::isfinite(mp.max_steer_rate) ? fmt(mp.max_steer_rate * rad2deg, 2) + " °/s" : "无")
            << "。<br>"
            << "<a href=\"effective_config.ini\">查看完整生效配置</a>。</div>";
        out << "<div class=\"notice\"><strong>本次实际应用的 MPC Q/R 数值范围</strong><br>";
        const char* html_weight_names[] = {"Qy", "Qpsi", "Qvy", "Qr", "Rsteer", "Rdelta"};
        for (std::size_t k = 0; k < 6; ++k)
            out << html_weight_names[k] << "：" << fmt(min_weights[k], 5)
                << "–" << fmt(max_weights[k], 5) << (k == 5 ? "" : "；");
        out << "<br>逐周期原值见 <a href=\"simulation.csv\">simulation.csv</a>。"
               "</div>";
        if (filtered_curvature)
            out << "<p>曲率图显示转向前馈所用的空间估计。CarSim 原始参考曲率和前馈曲率均保存在 simulation.csv。"
                   "横向误差沿用原生参考位置。</p>"
                   "<p><a href=\"06_native_reference_curvature.svg\">查看 CarSim 原生几何曲率</a></p>";
        out << "<div class=\"stats\">"
            << "<div class=\"stat\"><small>最大横向误差绝对值</small><strong>"
            << fmt(metrics.max_abs_y, 4) << " m</strong></div>"
            << "<div class=\"stat\"><small>横向误差平均绝对值</small><strong>"
            << fmt(metrics.mae_y, 4) << " m</strong></div>"
            << "<div class=\"stat\"><small>最大横摆角速度绝对值</small><strong>"
            << fmt(metrics.max_abs_yaw_rate_deg_s, 2) << " °/s</strong></div>"
            << "<div class=\"stat\"><small>横摆角速度平均绝对值</small><strong>"
            << fmt(metrics.mean_abs_yaw_rate_deg_s, 2) << " °/s</strong></div>"
            << "<div class=\"stat\"><small>最大实际前轮转角绝对值</small><strong>"
            << (metrics.has_front_wheel_angles ? fmt(metrics.max_abs_front_wheel_deg, 2) + " °" : "数据不可用")
            << "</strong></div>"
            << "<div class=\"stat\"><small>实际前轮转角平均绝对值</small><strong>"
            << (metrics.has_front_wheel_angles ? fmt(metrics.mean_abs_front_wheel_deg, 2) + " °" : "数据不可用")
            << "</strong></div>"
            << "<div class=\"stat\"><small>横向跟踪 RMSE</small><strong>"
            << fmt(metrics.rmse_y, 3) << " m</strong></div>"
            << "<div class=\"stat\"><small>航向角跟踪 RMSE</small><strong>"
            << fmt(metrics.rmse_heading_deg, 3) << "°</strong></div>"
            << "<div class=\"stat\"><small>最大前轮转角指令</small><strong>"
            << fmt(metrics.max_abs_steering_deg, 2) << "°</strong></div>"
            << "<div class=\"stat\"><small>最大转角指令变化率</small><strong>"
            << fmt(metrics.max_steering_rate_deg_s, 2) << " °/s</strong></div>"
            << "<div class=\"stat\"><small>CarSim 实测初速 / Np / Nc</small><strong>"
            << fmt(3.6 * rows.front().s.vx, 1) << " km/h / " << min_np;
        if (min_np != max_np) out << "–" << max_np;
        out << " / " << mp.nc << "</strong></div>";
        out << "<div class=\"stat\"><small>MPC 未正常求解步数</small><strong>"
            << metrics.qp_nonconverged_steps << "</strong></div>"
            << "<div class=\"stat\"><small>保持上一转角步数</small><strong>"
            << metrics.fallback_control_steps << "</strong></div>";
        out << "<div class=\"stat\"><small>单次优化平均耗时</small><strong>"
            << fmt(metrics.mean_qp_time_ms, 2) << " ms</strong></div></div>"
            << "<div class=\"grid\">";
        const char* plots[] = {"02_sideslip.svg", "05_steering.svg",
                                "02_lateral_velocity.svg", "02_yaw_rate.svg",
                                "02_heading.svg", "01_lateral_tracking.svg",
                                "03_lateral_error.svg", "04_heading_error.svg", "06_reference_curvature.svg",
                                "08_prediction_horizon.svg", "09_q_y_value.svg",
                                "09_q_psi_value.svg", "10_r_delta_value.svg"};
        for (const char* name : plots)
            out << "<div class=\"plot\"><img loading=\"lazy\" src=\""
                << name << "\" alt=\"" << name << "\"></div>";
        out << "</div></main><footer>原始数据：<a href=\"simulation.csv\">simulation.csv</a>"
            << " ｜ <a href=\"summary.txt\">统计摘要</a>"
            << " ｜ <a href=\"effective_config.ini\">本次生效参数</a>"
            << " ｜ <a href=\"tire_data.csv\">"
            << "C++ 预测模型轮胎数据</a><br>"
            << "车辆状态来自 CarSim；转角曲线是 C++ 控制指令，实际轮角尚未映射。MPC 使用三自由度动力学增量式 LTV 预测，沿预测轨迹逐步线性化并作 Tustin 离散化；质量、质心、偏航惯量、轮胎表和路面摩擦系数取自本次 CarSim Solver 回显。纵向轮胎滑移与刚度当前未映射，纵向加速度由相邻 CarSim 状态估计。"
            << "</footer></body></html>";
    }
    return metrics;
}
