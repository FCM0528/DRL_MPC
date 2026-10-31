#include "visualization.h"
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
    const double xaxismin = std::floor(xmin / xstep) * xstep;
    const double xaxismax = std::ceil(xmax / xstep) * xstep;
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
    for (const auto& row : samples) out.xy.emplace_back(getx(row), gety(row));
    return out;
}
void comparisonChart(const std::filesystem::path& file,
                     const std::string& title,
                     const std::string& xlabel,
                     const std::string& ylabel,
                     std::vector<Trace> traces,
                     bool integer_y = false) {
    for (auto& tr : traces) {
        if (tr.name == "Reference" || tr.name == "Zero") {
            tr.color = "#0f766e";
            tr.dash_pattern = "2,6";
            tr.stroke_width = 2.5;
        } else if (tr.name == "Fixed target") {
            tr.color = "#64748b";
            tr.dash_pattern = "8,5,2,5";
            tr.stroke_width = 2.2;
        } else if (tr.name == "DDPG target") {
            tr.color = "#d97706";
            tr.dash_pattern = "2,5,2,5,9,5";
            tr.stroke_width = 2.2;
        } else if (tr.name == "TD3 target") {
            tr.color = "#a78bfa";
            tr.dash_pattern = "2,6";
            tr.stroke_width = 2.2;
        } else if (tr.name.rfind("Fixed", 0) == 0) {
            tr.color = tr.name.find("ground") != std::string::npos ? "#60a5fa" : "#1d4ed8";
            tr.dash_pattern = tr.name.find("ground") != std::string::npos ? "8,5,2,5" : "15,8";
            tr.stroke_width = 3.3;
        } else if (tr.name.rfind("DDPG", 0) == 0) {
            tr.color = tr.name.find("ground") != std::string::npos ? "#ea580c" : "#be123c";
            tr.dash_pattern = tr.name.find("ground") != std::string::npos
                ? "4,4,2,4,2,4" : "10,5,2,5";
            tr.stroke_width = 3.2;
        } else if (tr.name.rfind("TD3", 0) == 0) {
            tr.color = tr.name.find("ground") != std::string::npos ? "#a78bfa" : "#6d28d9";
            tr.dash_pattern = tr.name.find("ground") != std::string::npos ? "3,4" : "5,4";
            tr.stroke_width = 3.2;
        }
    }
    // Draw the dashed fixed trajectory last so both controllers remain visible
    // where their measured trajectories coincide.
    const auto rank = [](const Trace& tr) {
        if (tr.name == "Reference" || tr.name == "Zero" ||
            tr.name.find("target") != std::string::npos) return 0;
        return tr.name.rfind("Fixed", 0) == 0 ? 2 : 1;
    };
    std::stable_sort(traces.begin(), traces.end(),
        [&](const Trace& a, const Trace& b) { return rank(a) < rank(b); });
    svgChart(file, title, xlabel, ylabel, traces, integer_y);
}
std::string fmt(double value, int precision = 4) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}
} // namespace

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
    double sum_y2 = 0.0, sum_heading2 = 0.0, sum_iters = 0.0;
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
        sum_y2 += p.lateral_error * p.lateral_error;
        sum_heading2 += p.heading_error * p.heading_error;
        metrics.max_abs_y = std::max(metrics.max_abs_y, std::abs(p.lateral_error));
        metrics.max_abs_heading_deg = std::max(metrics.max_abs_heading_deg,
                                               std::abs(p.heading_error * rad2deg));
        metrics.max_abs_steering_deg = std::max(metrics.max_abs_steering_deg,
                                                std::abs(p.steer * rad2deg));
        metrics.max_steering_rate_deg_s = std::max(metrics.max_steering_rate_deg_s,
                                         std::abs((p.steer - prev_delta) / mp.dt * rad2deg));
        if (p.is_command) {
            ++solver_samples;
            sum_iters += p.qp_iterations;
            sum_solve_ms += p.qp_solve_ms;
            metrics.max_qp_time_ms = std::max(metrics.max_qp_time_ms, p.qp_solve_ms);
            metrics.qp_nonconverged_steps +=
                (p.fallback_control || (p.qp_iterations > 0 && !p.qp_converged)) ? 1 : 0;
            metrics.grip_limited_steps += p.grip_limited_control ? 1 : 0;
            metrics.fallback_control_steps += p.fallback_control ? 1 : 0;
            metrics.peak_qp_primal_residual = std::max(metrics.peak_qp_primal_residual,
                                                      p.qp_primal_residual);
            metrics.peak_qp_dual_residual = std::max(metrics.peak_qp_dual_residual,
                                                    p.qp_dual_residual);
        }
        prev_delta = p.steer;
    }
    metrics.rmse_y = std::sqrt(sum_y2 / rows.size());
    metrics.rmse_heading_deg = rad2deg * std::sqrt(sum_heading2 / rows.size());
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
                   "grip_limited_control,q_y_applied,q_psi_applied,q_vy_applied,"
                   "q_r_applied,r_steer_applied,r_delta_applied,road_grip_estimate\n";
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
                    << p.prediction_horizon << ','
                    << (p.grip_limited_control ? 1 : 0);
                out << ',' << p.q_y << ',' << p.q_psi << ','
                    << p.q_vy << ',' << p.q_r << ','
                    << p.r_steer << ',' << p.r_delta << ',' << p.road_grip << '\n';
        }
    }

    auto xcoord = [](const SimulationRecord& v) { return v.s.x; };
    auto time = [](const SimulationRecord& v) { return v.time; };
    auto constant0 = [](const SimulationRecord&) { return 0.0; };
    const std::string blue = "#147bd1", orange = "#e96932", red = "#c83e58";
    svgChart(output / "01_lateral_tracking.svg", "Double lane change: lateral tracking",
             "Longitudinal position X (m)", "Lateral position Y (m)", {
             makeTrace(rows, "MPC actual", blue, xcoord,
                       [](const auto& v) {return v.s.y;}),
             makeTrace(rows, "Reference", orange, xcoord,
                       [](const auto& v) {return v.reference.y;}, true)});
    svgChart(output / "02_heading_tracking.svg", "Heading angle tracking",
             "Longitudinal position X (m)", "Heading angle (deg)", {
             makeTrace(rows, "MPC actual", blue, xcoord,
                       [](const auto& v) {return v.s.psi * rad2deg;}),
             makeTrace(rows, "Reference", orange, xcoord,
                       [](const auto& v) {return v.reference.psi * rad2deg;}, true)});
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
    svgChart(output / "05_steering.svg", "MPC front-wheel steering command",
             "Time (s)", "Front steering angle (deg)", {
             makeTrace(rows, "MPC steering", blue, time,
                       [](const auto& v) {return v.steer * rad2deg;})});
    svgChart(output / "06_yaw_rate.svg", "Yaw rate tracking",
             "Longitudinal position X (m)", "Yaw rate (deg/s)", {
             makeTrace(rows, "MPC actual", blue, xcoord,
                       [](const auto& v) {return v.s.r * rad2deg;}),
             makeTrace(rows, "Reference", orange, xcoord,
                       [](const auto& v) {return v.yaw_rate_ref * rad2deg;}, true)});
    std::vector<Trace> speed_traces{
        makeTrace(rows, "CarSim body Vx", blue, time,
                  [](const auto& v) {return 3.6 * v.s.vx;}),
        makeTrace(rows, "CarSim ground speed", "#5a8f50", time,
                  [](const auto& v) {return 3.6 * std::hypot(v.s.vx, v.s.vy);})};
    svgChart(output / "07_speed.svg", "Vehicle speed", "Time (s)", "Speed (km/h)", speed_traces);
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
    svgChart(output / "09_q_vy_value.svg", "Applied MPC Qvy",
             "Longitudinal position X (m)", "Qvy absolute value", {
             makeTrace(rows, "Qvy", blue, xcoord, [](const auto& v){return v.q_vy;})});
    svgChart(output / "09_q_r_value.svg", "Applied MPC Qr",
             "Longitudinal position X (m)", "Qr absolute value", {
             makeTrace(rows, "Qr", blue, xcoord, [](const auto& v){return v.q_r;})});
    svgChart(output / "10_r_steer_value.svg", "Applied MPC Rsteer",
             "Longitudinal position X (m)", "Rsteer absolute value", {
             makeTrace(rows, "Rsteer", blue, xcoord, [](const auto& v){return v.r_steer;})});
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
            << "Controller mode: " << (outcome.controller_mode == "eval"
                ? (outcome.rl_algorithm == "td3" ? "TD3" : "DDPG") : "fixed")
            << (outcome.controller_mode == "eval" ? " adaptive prediction horizon and Q/R weights" : " prediction horizon and Q/R weights") << '\n'
            << (outcome.controller_mode == "eval"
                ? (outcome.rl_algorithm == "td3" ? "TD3" : "DDPG") + std::string(" policy file: ") + outcome.policy_file.string() + "\n" : "")
            << "Run status: " << (outcome.completed ? "completed" : "incomplete") << '\n'
            << "Requested distance (m): " << fmt(outcome.requested_distance_m, 3) << '\n'
            << "Stop reason: " << (outcome.completed ? "none" : outcome.stop_reason) << '\n'
            << "First MPC fallback reason: "
            << (outcome.first_fallback_reason.empty() ? "none" : outcome.first_fallback_reason)
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
                   "CarSim Visualizer source: carsim_native/Run_visualizer.par (unmodified generated Parsfile)\n"
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
        out << "MPC prediction states: [vy,vx,psi,r,Y,X]; three-DOF dynamic model only\n"
                << "MPC augmentation: previous steering state, optimized steering increments\n"
                << "MPC discretization: stage-varying CarSim operating points, bilinear Tustin\n"
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
                << (mp.nonlinear_tire ?
                    "MPC approximation: static axle loads; nonlinear tire table at each operating point with a slope floor; CarSim plant remains nonlinear\n" :
                    "MPC approximation: static axle loads and small-slip linearization; CarSim plant remains nonlinear\n");
        out
            << "Configured fixed-mode Q diag: " << fmt(mp.q_y) << ", " << fmt(mp.q_psi)
            << ", " << fmt(mp.q_vy) << ", " << fmt(mp.q_r) << '\n'
            << "Configured fixed-mode R steering feedforward deviation via increments: " << fmt(mp.r_steer) << '\n'
            << "Configured fixed-mode R_delta increment penalty: " << fmt(mp.r_delta) << '\n'
            << "Terminal multiplier: " << fmt(mp.terminal_scale) << '\n'
            << "Steering limit (deg): " << fmt(mp.max_steer * rad2deg) << '\n'
            << "Steering slew limit (deg/s): " << fmt(mp.max_steer_rate * rad2deg) << '\n'
            << "Speed-based slew schedule: " << (mp.speed_rate_enabled ? "enabled" : "disabled")
            << "; start/end (km/h): " << fmt(mp.speed_rate_start * 3.6) << " / " << fmt(mp.speed_rate_end * 3.6)
            << "; end limit (deg/s): " << fmt(mp.speed_rate_end_limit * rad2deg) << '\n'
            << "Simulation samples: " << rows.size() << '\n'
            << "Distance reached (m): " << fmt(rows.back().s.x, 3) << '\n'
            << "Reference parameters: CarSim LTARG_TABLE in staged Run_all.par\n"
            << "RMSE lateral (m): " << fmt(metrics.rmse_y, 6) << '\n'
            << "Max absolute lateral error (m): " << fmt(metrics.max_abs_y, 6) << '\n'
            << "RMSE heading (deg): " << fmt(metrics.rmse_heading_deg, 6) << '\n'
            << "Max absolute heading error (deg): " << fmt(metrics.max_abs_heading_deg, 6) << '\n'
            << "Max absolute front steering command (deg): "
            << fmt(metrics.max_abs_steering_deg, 6) << '\n'
            << "Max front steering command slew (deg/s): "
            << fmt(metrics.max_steering_rate_deg_s, 6) << '\n';
        out
            << "Average QP iterations per step: " << fmt(metrics.mean_qp_iterations, 2) << '\n'
            << "Mean QP solve time (ms): " << fmt(metrics.mean_qp_time_ms, 4) << '\n'
            << "Peak QP solve time (ms): " << fmt(metrics.max_qp_time_ms, 4) << '\n'
            << "Nonconverged QP steps: " << metrics.qp_nonconverged_steps << '\n'
            << "Note: includes attempted nonconverged QPs and fallback steering; excludes grip tracking without a QP attempt.\n"
            << "Grip-limited reference tracking steps: " << metrics.grip_limited_steps << '\n'
            << "Fallback steering steps: " << metrics.fallback_control_steps << '\n';
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
            << (outcome.controller_mode == "eval"
                ? (outcome.rl_algorithm == "td3" ? " · TD3 自适应预测时域与 Q/R 权重 MPC 双移线轨迹跟踪</h1>" : " · DDPG 自适应预测时域与 Q/R 权重 MPC 双移线轨迹跟踪</h1>")
                : " · 固定预测时域 MPC 双移线轨迹跟踪</h1>")
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
                    << " km/h（经本次 CarSim Solver 回显确认；实际车速见下图）<br>";
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
        } else if (metrics.fallback_control_steps > 0) {
            out << "<div class=\"notice\">MPC 曾求解失败，后续启用了受转角速率约束的后备转向。"
                << "这些区间不代表 MPC 正常跟踪。</div>";
        }
        if (metrics.grip_limited_steps > 0)
            out << "<div class=\"notice\">控制器假设的轮胎力上限触发了受限转向控制，共 "
                << metrics.grip_limited_steps
                << " 个周期。这是 C++ 控制器模型的判断，不代表 CarSim 实际轮胎达到极限。"
                << "这些周期未求解 QP，已与求解失败单独统计。</div>";
        out << "<div class=\"notice\"><strong>本次实际应用的 MPC Q/R 数值范围</strong><br>";
        const char* html_weight_names[] = {"Qy", "Qpsi", "Qvy", "Qr", "Rsteer", "Rdelta"};
        for (std::size_t k = 0; k < 6; ++k)
            out << html_weight_names[k] << "：" << fmt(min_weights[k], 5)
                << "–" << fmt(max_weights[k], 5) << (k == 5 ? "" : "；");
        out << "<br>逐周期原值见 <a href=\"simulation.csv\">simulation.csv</a>。"
               "</div>";
        out << "<div class=\"stats\">"
            << "<div class=\"stat\"><small>横向跟踪 RMSE</small><strong>"
            << fmt(metrics.rmse_y, 3) << " m</strong></div>"
            << "<div class=\"stat\"><small>横向误差最大绝对值</small><strong>"
            << fmt(metrics.max_abs_y, 3) << " m</strong></div>"
            << "<div class=\"stat\"><small>航向角跟踪 RMSE</small><strong>"
            << fmt(metrics.rmse_heading_deg, 3) << "°</strong></div>"
            << "<div class=\"stat\"><small>最大前轮转角指令</small><strong>"
            << fmt(metrics.max_abs_steering_deg, 2) << "°</strong></div>"
            << "<div class=\"stat\"><small>CarSim 实测初速 / Np / Nc</small><strong>"
            << fmt(3.6 * rows.front().s.vx, 1) << " km/h / " << min_np;
        if (min_np != max_np) out << "–" << max_np;
        out << " / " << mp.nc << "</strong></div>";
        out << "<div class=\"stat\"><small>MPC 未正常求解步数</small><strong>"
            << metrics.qp_nonconverged_steps << "</strong></div>"
            << "<div class=\"stat\"><small>C++ 控制器受限模式步数</small><strong>"
            << metrics.grip_limited_steps << "</strong></div>"
            << "<div class=\"stat\"><small>后备转向步数</small><strong>"
            << metrics.fallback_control_steps << "</strong></div>";
        out << "<div class=\"stat\"><small>单次优化平均耗时</small><strong>"
            << fmt(metrics.mean_qp_time_ms, 2) << " ms</strong></div></div>"
            << "<div class=\"grid\">";
        const char* plots[] = {"01_lateral_tracking.svg", "02_heading_tracking.svg",
                                "03_lateral_error.svg", "04_heading_error.svg",
                                "05_steering.svg", "06_yaw_rate.svg", "07_speed.svg",
                                "08_prediction_horizon.svg", "09_q_y_value.svg",
                                "09_q_psi_value.svg", "09_q_vy_value.svg",
                                "09_q_r_value.svg", "10_r_steer_value.svg",
                                "10_r_delta_value.svg"};
        for (const char* name : plots)
            out << "<div class=\"plot\"><img loading=\"lazy\" src=\""
                << name << "\" alt=\"" << name << "\"></div>";
        out << "</div></main><footer>原始数据：<a href=\"simulation.csv\">simulation.csv</a>"
            << " ｜ <a href=\"summary.txt\">统计摘要</a>"
            << " ｜ <a href=\"effective_config.ini\">本次生效参数</a>"
            << " ｜ <a href=\"tire_data.csv\">"
            << "C++ 预测模型轮胎数据</a><br>"
            << "车辆状态来自 CarSim；转角曲线是 C++ 控制指令，实际轮角尚未映射。MPC 使用三自由度动力学增量式 LTV 预测，沿预测轨迹逐步线性化并作 Tustin 离散化；质量、质心、偏航惯量、轮胎表和路面摩擦系数取自本次 CarSim Solver 回显。纵向轮胎滑移与刚度当前未映射，纵向加速度由相邻 CarSim 状态估计。DDPG 在所有车速均调节 Np/Q/R。"
            << "</footer></body></html>";
    }
    return metrics;
}

void exportComparisonVisuals(const std::filesystem::path& output,
    const std::vector<SimulationRecord>& fixed,
    const std::vector<SimulationRecord>& adaptive) {
    if (fixed.empty() || adaptive.empty())
        throw std::invalid_argument("Controller comparison requires both trajectories");
    std::filesystem::create_directories(output);
    const auto x = [](const auto& v) { return v.s.x; };
    const auto t = [](const auto& v) { return v.time; };
    const auto zero = [](const auto&) { return 0.0; };
    const std::string fixed_color = "#2563eb";
    const std::string adaptive_color = "#dc2626";
    const std::string reference_color = "#e96932";
    comparisonChart(output / "01_lateral_tracking.svg", "Double lane change: fixed Np vs DDPG",
        "Longitudinal position X (m)", "Lateral position Y (m)", {
        makeTrace(fixed, "Reference", reference_color, x,
            [](const auto& v) { return v.reference.y; }, true),
        makeTrace(fixed, "Fixed Np", fixed_color, x,
            [](const auto& v) { return v.s.y; }),
        makeTrace(adaptive, "DDPG Np", adaptive_color, x,
            [](const auto& v) { return v.s.y; })});
    comparisonChart(output / "02_heading_tracking.svg", "Heading angle: fixed Np vs DDPG",
        "Longitudinal position X (m)", "Heading angle (deg)", {
        makeTrace(fixed, "Reference", reference_color, x,
            [](const auto& v) { return v.reference.psi * rad2deg; }, true),
        makeTrace(fixed, "Fixed Np", fixed_color, x,
            [](const auto& v) { return v.s.psi * rad2deg; }),
        makeTrace(adaptive, "DDPG Np", adaptive_color, x,
            [](const auto& v) { return v.s.psi * rad2deg; })});
    comparisonChart(output / "03_lateral_error.svg", "Lateral error: fixed Np vs DDPG",
        "Longitudinal position X (m)", "Y - Yref (m)", {
        makeTrace(fixed, "Zero", "#a4b0bf", x, zero, true),
        makeTrace(fixed, "Fixed Np", fixed_color, x,
            [](const auto& v) { return v.lateral_error; }),
        makeTrace(adaptive, "DDPG Np", adaptive_color, x,
            [](const auto& v) { return v.lateral_error; })});
    comparisonChart(output / "04_heading_error.svg", "Heading error: fixed Np vs DDPG",
        "Longitudinal position X (m)", "Psi - Psiref (deg)", {
        makeTrace(fixed, "Zero", "#a4b0bf", x, zero, true),
        makeTrace(fixed, "Fixed Np", fixed_color, x,
            [](const auto& v) { return v.heading_error * rad2deg; }),
        makeTrace(adaptive, "DDPG Np", adaptive_color, x,
            [](const auto& v) { return v.heading_error * rad2deg; })});
    comparisonChart(output / "05_steering.svg", "Steering: fixed Np vs DDPG",
        "Time (s)", "Front steering angle (deg)", {
        makeTrace(fixed, "Fixed Np", fixed_color, t,
            [](const auto& v) { return v.steer * rad2deg; }),
        makeTrace(adaptive, "DDPG Np", adaptive_color, t,
            [](const auto& v) { return v.steer * rad2deg; })});
    comparisonChart(output / "06_yaw_rate.svg", "Yaw rate: fixed Np vs DDPG",
        "Longitudinal position X (m)", "Yaw rate (deg/s)", {
        makeTrace(fixed, "Fixed target", reference_color, x,
            [](const auto& v) { return v.yaw_rate_ref * rad2deg; }, true),
        makeTrace(adaptive, "DDPG target", "#f59e0b", x,
            [](const auto& v) { return v.yaw_rate_ref * rad2deg; }, true),
        makeTrace(fixed, "Fixed Np", fixed_color, x,
            [](const auto& v) { return v.s.r * rad2deg; }),
        makeTrace(adaptive, "DDPG Np", adaptive_color, x,
            [](const auto& v) { return v.s.r * rad2deg; })});
    comparisonChart(output / "07_speed.svg", "Vehicle speed: fixed Np vs DDPG",
        "Time (s)", "Speed (km/h)", {
        makeTrace(fixed, "Fixed vx", fixed_color, t,
            [](const auto& v) { return 3.6 * v.s.vx; }),
        makeTrace(adaptive, "DDPG vx", adaptive_color, t,
            [](const auto& v) { return 3.6 * v.s.vx; }),
        makeTrace(fixed, "Fixed ground", "#60a5fa", t,
            [](const auto& v) { return 3.6 * std::hypot(v.s.vx, v.s.vy); }, true),
        makeTrace(adaptive, "DDPG ground", "#f87171", t,
            [](const auto& v) { return 3.6 * std::hypot(v.s.vx, v.s.vy); }, true)});
    comparisonChart(output / "08_prediction_horizon.svg", "Prediction horizon: fixed Np vs DDPG",
        "Longitudinal position X (m)", "Np (steps)", {
        makeTrace(fixed, "Fixed Np", fixed_color, x,
            [](const auto& v) { return static_cast<double>(v.prediction_horizon); }),
        makeTrace(adaptive, "DDPG Np", adaptive_color, x,
            [](const auto& v) { return static_cast<double>(v.prediction_horizon); })}, true);
}

void exportThreeComparisonVisuals(const std::filesystem::path& output,
    const std::vector<SimulationRecord>& fixed,
    const std::vector<SimulationRecord>& ddpg,
    const std::vector<SimulationRecord>& td3) {
    if (fixed.empty() || ddpg.empty() || td3.empty())
        throw std::invalid_argument("Three-controller comparison requires all trajectories");
    std::filesystem::create_directories(output);
    const auto x = [](const auto& v) { return v.s.x; };
    const auto t = [](const auto& v) { return v.time; };
    const auto zero = [](const auto&) { return 0.0; };
    const auto trace = [&](const auto& rows, const char* name,
                           const auto& getx, const auto& gety) {
        return makeTrace(rows, name, "#2563eb", getx, gety);
    };
    const auto draw = [&](const char* file, const char* title,
                          const char* xlabel, const char* ylabel,
                          const auto& getx, const auto& gety,
                          std::vector<Trace> extra, bool integer_y = false) {
        extra.push_back(trace(fixed, "Fixed MPC", getx, gety));
        extra.push_back(trace(ddpg, "DDPG", getx, gety));
        extra.push_back(trace(td3, "TD3", getx, gety));
        comparisonChart(output / file, title, xlabel, ylabel, std::move(extra), integer_y);
    };
    const char* xpos = "Longitudinal position X (m)";
    draw("01_lateral_tracking.svg", "Lateral path tracking", xpos, "Lateral position Y (m)", x,
        [](const auto& v) { return v.s.y; },
        {makeTrace(fixed, "Reference", "#0f766e", x, [](const auto& v) { return v.reference.y; }, true)});
    draw("02_heading_tracking.svg", "Heading angle tracking", xpos, "Heading angle (deg)", x,
        [](const auto& v) { return v.s.psi * rad2deg; },
        {makeTrace(fixed, "Reference", "#0f766e", x, [](const auto& v) { return v.reference.psi * rad2deg; }, true)});
    draw("03_lateral_error.svg", "Lateral tracking error", xpos, "Y - Yref (m)", x,
        [](const auto& v) { return v.lateral_error; },
        {makeTrace(fixed, "Zero", "#0f766e", x, zero, true)});
    draw("04_heading_error.svg", "Heading tracking error", xpos, "Psi - Psiref (deg)", x,
        [](const auto& v) { return v.heading_error * rad2deg; },
        {makeTrace(fixed, "Zero", "#0f766e", x, zero, true)});
    draw("05_steering.svg", "Front-wheel steering command", "Time (s)", "Steering angle (deg)", t,
        [](const auto& v) { return v.steer * rad2deg; }, {});
    draw("06_yaw_rate.svg", "Yaw rate", xpos, "Yaw rate (deg/s)", x,
        [](const auto& v) { return v.s.r * rad2deg; },
        {makeTrace(fixed, "Fixed target", "#64748b", x, [](const auto& v) { return v.yaw_rate_ref * rad2deg; }, true),
         makeTrace(ddpg, "DDPG target", "#d97706", x, [](const auto& v) { return v.yaw_rate_ref * rad2deg; }, true),
         makeTrace(td3, "TD3 target", "#a78bfa", x, [](const auto& v) { return v.yaw_rate_ref * rad2deg; }, true)});
    comparisonChart(output / "07_speed.svg", "Vehicle speed", "Time (s)", "Speed (km/h)", {
        trace(fixed, "Fixed vx", t, [](const auto& v) { return 3.6 * v.s.vx; }),
        trace(ddpg, "DDPG vx", t, [](const auto& v) { return 3.6 * v.s.vx; }),
        trace(td3, "TD3 vx", t, [](const auto& v) { return 3.6 * v.s.vx; }),
        trace(fixed, "Fixed ground", t, [](const auto& v) { return 3.6 * std::hypot(v.s.vx, v.s.vy); }),
        trace(ddpg, "DDPG ground", t, [](const auto& v) { return 3.6 * std::hypot(v.s.vx, v.s.vy); }),
        trace(td3, "TD3 ground", t, [](const auto& v) { return 3.6 * std::hypot(v.s.vx, v.s.vy); })});
    draw("08_prediction_horizon.svg", "Prediction horizon", xpos, "Np (steps)", x,
        [](const auto& v) { return static_cast<double>(v.prediction_horizon); }, {}, true);
    draw("11_road_grip.svg", "Road grip from CarSim map at CG station", xpos, "Road grip mu", x,
        [](const auto& v) { return v.road_grip; }, {});
    struct WeightChart { const char* file; const char* title; double SimulationRecord::*field; };
    const std::array<WeightChart, 6> weights{{
        {"09_q_y_value.svg", "Lateral-error weight Qy", &SimulationRecord::q_y},
        {"09_q_psi_value.svg", "Heading-error weight Qpsi", &SimulationRecord::q_psi},
        {"09_q_vy_value.svg", "Lateral-velocity weight Qvy", &SimulationRecord::q_vy},
        {"09_q_r_value.svg", "Yaw-rate weight Qr", &SimulationRecord::q_r},
        {"10_r_steer_value.svg", "Steering weight Rsteer", &SimulationRecord::r_steer},
        {"10_r_delta_value.svg", "Steering-increment weight Rdelta", &SimulationRecord::r_delta}}};
    for (const auto& weight : weights)
        draw(weight.file, weight.title, xpos, "Applied weight", x,
             [field = weight.field](const auto& v) { return v.*field; }, {});
}
