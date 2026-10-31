#include "carsim_plant.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
std::string trimSimfileValue(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string expandSimfileMacros(std::string value,
                                const std::map<std::string, std::string>& macros) {
    for (int pass = 0; pass < 8; ++pass) {
        bool changed = false;
        for (const auto& [name, replacement] : macros) {
            std::size_t pos = 0;
            while ((pos = value.find(name, pos)) != std::string::npos) {
                value.replace(pos, name.size(), replacement);
                pos += replacement.size();
                changed = true;
            }
        }
        if (!changed) break;
    }
    if (value.find("$(") != std::string::npos)
        throw std::runtime_error("Unresolved macro in CarSim Simfile: " + value);
    return value;
}

bool sameFileContents(const std::filesystem::path& left,
                      const std::filesystem::path& right) {
    if (!std::filesystem::is_regular_file(left) ||
        !std::filesystem::is_regular_file(right)) return false;
    if (std::filesystem::file_size(left) != std::filesystem::file_size(right)) return false;
    std::ifstream a(left, std::ios::binary), b(right, std::ios::binary);
    if (!a || !b) throw std::runtime_error("Cannot compare CarSim run snapshots");
    return std::equal(std::istreambuf_iterator<char>(a), std::istreambuf_iterator<char>(),
                      std::istreambuf_iterator<char>(b), std::istreambuf_iterator<char>());
}
}  // namespace

void extendCarSimStationStop(const std::filesystem::path& parsfile, double distance_m) {
    if (!std::isfinite(distance_m) || distance_m <= 0.0)
        throw std::invalid_argument("Invalid requested CarSim distance");
    std::ifstream input(parsfile, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read staged CarSim stop settings");
    std::vector<std::string> lines;
    int opt_count=0, stop_count=0, opt=0;
    std::size_t stop_line=0;
    double stop=0.0;
    std::string line;
    while (std::getline(input,line)) {
        std::istringstream fields(line); std::string keyword; fields >> keyword;
        if(keyword=="OPT_STOP") {
            if (!(fields >> opt)) throw std::runtime_error("Invalid CarSim OPT_STOP");
            ++opt_count;
        } else if(keyword=="SSTOP") {
            ++stop_count; stop_line=lines.size();
            if (!(fields >> stop)) stop=std::numeric_limits<double>::quiet_NaN();
        }
        lines.push_back(line);
    }
    input.close();
    if (opt_count==0) return;
    if (opt_count!=1) throw std::runtime_error("Multiple CarSim OPT_STOP entries are unsupported");
    if (opt==0) return;
    if (opt!=1 || stop_count!=1 || !std::isfinite(stop))
        throw std::runtime_error("Unsupported CarSim station-stop configuration; numeric SSTOP required");
    const double minimum=distance_m+10.0;
    if (stop>=minimum) return;
    std::ostringstream replacement; replacement << std::setprecision(17) << "SSTOP " << minimum;
    lines[stop_line]=replacement.str();
    std::ofstream output(parsfile,std::ios::binary|std::ios::trunc);
    if(!output) throw std::runtime_error("Cannot update staged CarSim station limit");
    for(const auto& item:lines) output << item << '\n';
    if(!output) throw std::runtime_error("Cannot finish staged CarSim station limit");
}

CarSimRunInfo readCarSimRunInfo(const std::filesystem::path& parsfile) {
    std::ifstream input(parsfile);
    if (!input) throw std::runtime_error("Cannot read CarSim run snapshot: " + parsfile.string());
    CarSimRunInfo info;
    int speed_target_count = 0, stop_time_count = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("#FullDataName ", 0) == 0) {
            const auto first = line.find('`');
            const auto second = first == std::string::npos ? first : line.find('`', first + 1);
            if (first == std::string::npos || second == std::string::npos) continue;
            const auto category = line.substr(14, first - 14);
            const auto title = line.substr(first + 1, second - first - 1);
            if (category == "Vehicle: Assembly" && info.vehicle_dataset.empty())
                info.vehicle_dataset = title;
            else if (category == "Road: 3D Surface (All Properties)" && info.road_dataset.empty())
                info.road_dataset = title;
            else if (category == "Procedures" && info.procedure_dataset.empty())
                info.procedure_dataset = title;
            else if (category == "Control: Steering by the Closed-loop Driver Model" &&
                     info.reference_dataset.empty())
                info.reference_dataset = title;
            continue;
        }
        std::istringstream fields(line);
        std::string keyword;
        fields >> keyword;
        if (keyword == "SPEED_TARGET_CONSTANT") {
            std::string value;
            fields >> value;
            if (++speed_target_count == 1) info.speed_target_constant = value;
            else info.speed_target_constant.clear();
        } else if (keyword == "TSTOP") {
            std::string value;
            fields >> value;
            if (++stop_time_count == 1) info.stop_time = value;
            else info.stop_time.clear();
        }
    }
    return info;
}

double CarSimMpcModel::RoadGrip::at(double station) const {
    double base = constant;
    if (!table.empty()) {
        const double s = (station - station_start) / station_scale;
        auto next = std::upper_bound(table.begin(), table.end(), s,
            [](double value, const auto& point) { return value < point.first; });
        base = next == table.begin() ? table.front().second : std::prev(next)->second;
    }
    const double grip = base * gain + offset + lateral;
    if (!std::isfinite(grip) || grip <= 0.0)
        throw std::runtime_error("Invalid effective CarSim road grip");
    return grip;
}

CarSimMpcModel readCarSimMpcModel(const std::filesystem::path& native_dir) {
    std::ifstream in(native_dir / "LastRun_echo.par");
    if (!in) throw std::runtime_error("Current CarSim solver echo is unavailable");
    std::map<std::string, double> values;
    std::string speed_target_kmh;
    bool road_combine_seen = false;
    std::map<std::string, int> tire_modifiers_seen;
    std::array<std::vector<std::vector<double>>, 4> carpets;
    int active = -1;
    bool road_table_active = false;
    std::vector<std::pair<double, double>> road_table;
    std::string line;
    while (std::getline(in, line)) {
        auto body = trimSimfileValue(line);
        if (body.empty()) continue;
        if (road_table_active) {
            if (body == "ENDTABLE") { road_table_active = false; continue; }
            std::istringstream fields(body);
            double station, mu; char comma;
            if (!(fields >> station >> comma >> mu) || comma != ',' ||
                !std::isfinite(station) || !std::isfinite(mu) ||
                (!road_table.empty() && station <= road_table.back().first))
                throw std::runtime_error("Invalid CarSim MU_ROAD_TABLE row: " + body);
            road_table.emplace_back(station, mu);
            continue;
        }
        if (active >= 0) {
            if (body == "ENDTABLE") { active = -1; continue; }
            std::vector<double> row;
            std::istringstream fields(body);
            std::string field;
            while (std::getline(fields, field, ',')) {
                field = trimSimfileValue(field);
                std::size_t used = 0;
                const double number = std::stod(field, &used);
                if (used != field.size() || !std::isfinite(number))
                    throw std::runtime_error("Invalid CarSim Fy tire carpet");
                row.push_back(number);
            }
            carpets[active].push_back(std::move(row));
            continue;
        }
        if (body.front() == '!') body = trimSimfileValue(body.substr(1));
        std::istringstream fields(body);
        std::string key, raw;
        fields >> key >> raw;
        if (key.empty() || raw.empty()) continue;
        if (key == "MU_ROAD_TABLE(1)") {
            if (raw != "STEP" || !road_table.empty())
                throw std::runtime_error("Only one CarSim MU_ROAD_TABLE STEP is supported");
            road_table_active = true;
            continue;
        }
        if (key.rfind("MU_ROAD_TABLE", 0) == 0 || key.rfind("MU_ROAD_CARPET", 0) == 0 ||
            key.rfind("MU_ROAD_L_TABLE", 0) == 0 ||
            key == "MU_ROAD_CONSTANT(2)")
            throw std::runtime_error("Variable or multiple CarSim road friction surfaces require a dynamic MPC model");
        if (key == "SPEED_TARGET_CONSTANT(1)" &&
            body.find("; km/h") != std::string::npos) speed_target_kmh = raw;
        for (int axle = 1; axle <= 2; ++axle)
            for (int side = 1; side <= 2; ++side) {
                const std::string suffix = "(" + std::to_string(axle) + "," +
                                           std::to_string(side) + ")";
                if (key == "FY_TIRE_CARPET" + suffix) {
                    if (raw != "2D_FROM_ZERO")
                        throw std::runtime_error("Unsupported CarSim Fy tire model");
                    active = 2 * (axle - 1) + side - 1;
                }
                for (const char* prefix : {"FY_TIRE_GAIN", "FY_TIRE_OFFSET",
                     "ALPHA_START_FY_TIRE", "ALPHA_SCALE_FY_TIRE",
                     "FZ_START_FY_TIRE", "FZ_SCALE_FY_TIRE", "MU_REF_Y"}) {
                    if (key != std::string(prefix) + suffix) continue;
                    const bool unity = std::string(prefix) == "FY_TIRE_GAIN" ||
                        std::string(prefix) == "ALPHA_SCALE_FY_TIRE" ||
                        std::string(prefix) == "FZ_SCALE_FY_TIRE" ||
                        std::string(prefix) == "MU_REF_Y";
                    if (std::stod(raw) != (unity ? 1.0 : 0.0))
                        throw std::runtime_error("Unsupported CarSim tire modifier: " + key);
                    ++tire_modifiers_seen[key];
                }
            }
        if (key == "MU_ROAD_COMBINE(1)") {
            if (raw != "ADD")
                throw std::runtime_error("Unsupported CarSim road friction combination");
            road_combine_seen = true;
        }
        for (const char* wanted : {"M_TL", "IZZ_TL", "LX_CG_TL", "LX_AXLE(1)",
             "LX_AXLE(2)", "MU_ROAD_CONSTANT(1)", "MU_ROAD_GAIN(1)",
             "MU_ROAD_OFFSET(1)", "MU_ROAD_L_CONSTANT(1)", "SSTART_MU_ROAD(1)",
             "SSCALE_MU_ROAD(1)", "ROAD_PATH_ID(1)", "PATH_ID_DM"})
            if (key == wanted) values[key] = std::stod(raw);
    }
    const auto get = [&](const char* key) {
        const auto it = values.find(key);
        if (it == values.end() || !std::isfinite(it->second))
            throw std::runtime_error(std::string("CarSim echo missing ") + key);
        return it->second;
    };
    CarSimMpcModel model;
    if (!road_combine_seen)
        throw std::runtime_error("CarSim echo missing MU_ROAD_COMBINE(1)");
    for (int axle = 1; axle <= 2; ++axle)
        for (int side = 1; side <= 2; ++side) {
            const std::string suffix = "(" + std::to_string(axle) + "," +
                                       std::to_string(side) + ")";
            for (const char* prefix : {"FY_TIRE_GAIN", "FY_TIRE_OFFSET",
                 "ALPHA_START_FY_TIRE", "ALPHA_SCALE_FY_TIRE",
                 "FZ_START_FY_TIRE", "FZ_SCALE_FY_TIRE", "MU_REF_Y"})
                if (tire_modifiers_seen[std::string(prefix) + suffix] != 1)
                    throw std::runtime_error("CarSim echo missing tire modifier: " +
                                             std::string(prefix) + suffix);
        }
    model.speed_target_kmh = speed_target_kmh;
    model.vehicle.mass = get("M_TL");
    model.vehicle.iz = get("IZZ_TL");
    model.vehicle.a = (get("LX_CG_TL") - get("LX_AXLE(1)")) / 1000.0;
    model.vehicle.b = (get("LX_AXLE(2)") - get("LX_CG_TL")) / 1000.0;
    if (road_table_active) throw std::runtime_error("Unterminated CarSim MU_ROAD_TABLE");
    model.road_grip.table = std::move(road_table);
    model.road_grip.gain = get("MU_ROAD_GAIN(1)");
    model.road_grip.offset = get("MU_ROAD_OFFSET(1)");
    model.road_grip.lateral = get("MU_ROAD_L_CONSTANT(1)");
    if (model.road_grip.variable()) {
        if (get("ROAD_PATH_ID(1)") != get("PATH_ID_DM"))
            throw std::runtime_error("Road friction station uses a different reference path than the driver");
        model.road_grip.station_start = get("SSTART_MU_ROAD(1)");
        model.road_grip.station_scale = get("SSCALE_MU_ROAD(1)");
        if (!(model.road_grip.station_scale > 0.0))
            throw std::runtime_error("Invalid SSCALE_MU_ROAD(1)");
        for (const auto& point : model.road_grip.table)
            (void)model.road_grip.at(point.first * model.road_grip.station_scale + model.road_grip.station_start);
    } else model.road_grip.constant = get("MU_ROAD_CONSTANT(1)");
    model.vehicle.road_grip_scale = model.road_grip.at(0.0);
    if (!(model.vehicle.mass > 0 && model.vehicle.iz > 0 &&
          model.vehicle.a > 0 && model.vehicle.b > 0 &&
          model.vehicle.road_grip_scale > 0))
        throw std::runtime_error("Invalid CarSim vehicle mass, geometry, inertia or grip");
    const auto& table = carpets.front();
    if (table.size() < 3 || table.front().size() < 3 || table.front().front() != 0)
        throw std::runtime_error("CarSim Fy tire carpet is missing or incomplete");
    for (const auto& carpet : carpets)
        if (carpet != table)
            throw std::runtime_error("CarSim front/rear tires differ; axle-specific tables required");
    model.tire_csv = native_dir / "carsim_tire_fy.csv";
    std::ofstream out(model.tire_csv, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write CarSim tire snapshot");
    out << "slip_deg" << std::setprecision(15);
    for (std::size_t j = 1; j < table.front().size(); ++j) out << ',' << table.front()[j];
    out << '\n';
    for (std::size_t i = 1; i < table.size(); ++i) {
        if (table[i].size() != table.front().size())
            throw std::runtime_error("CarSim tire carpet row width differs");
        for (std::size_t j = 0; j < table[i].size(); ++j)
            out << (j ? "," : "") << table[i][j];
        out << '\n';
    }
    out.close();
    TireTable tires(model.tire_csv);
    const auto [front_load, rear_load] = staticWheelLoads(model.vehicle);
    model.front_stiffness_n_per_rad = 2.0 * model.vehicle.road_grip_scale *
        tires.corneringStiffness(front_load);
    model.rear_stiffness_n_per_rad = 2.0 * model.vehicle.road_grip_scale *
        tires.corneringStiffness(rear_load);
    return model;
}

std::filesystem::path stageCarSimSimfile(const std::filesystem::path& source,
                                         const std::filesystem::path& output_dir) {
    std::ifstream input(source);
    if (!input) throw std::runtime_error("Cannot open CarSim Simfile: " + source.string());
    std::vector<std::string> lines;
    std::map<std::string, std::string> macros;
    std::filesystem::path data_dir;
    std::filesystem::path program_dir;
    std::string input_value;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
        std::istringstream fields(line);
        std::string keyword;
        fields >> keyword;
        if (keyword == "SET_MACRO") {
            std::string name;
            fields >> name;
            std::string value;
            std::getline(fields, value);
            macros[name] = trimSimfileValue(value);
        } else if (keyword == "DATADIR") {
            std::string value;
            std::getline(fields, value);
            data_dir = trimSimfileValue(value);
        } else if (keyword == "PROGDIR") {
            std::string value;
            std::getline(fields, value);
            program_dir = trimSimfileValue(value);
        } else if (keyword == "INPUT") {
            if (!input_value.empty())
                throw std::runtime_error("CarSim Simfile has multiple INPUT entries");
            std::getline(fields, input_value);
            input_value = trimSimfileValue(input_value);
        }
    }
    if (input_value.empty()) throw std::runtime_error("CarSim Simfile has no INPUT Parsfile");
    std::filesystem::path parsfile = expandSimfileMacros(input_value, macros);
    if (parsfile.is_relative()) {
        if (data_dir.empty()) data_dir = std::filesystem::absolute(source).parent_path();
        parsfile = data_dir / parsfile;
    }
    parsfile = std::filesystem::absolute(parsfile).lexically_normal();
    if (!std::filesystem::is_regular_file(parsfile))
        throw std::runtime_error("CarSim Parsfile does not exist: " + parsfile.string());

    const auto dir = std::filesystem::absolute(output_dir).lexically_normal();
    std::filesystem::create_directories(dir);
    // Keep the complete generated run data for VS Visualizer. The solver-only
    // copy below drops unsupported SDK EXPORT channels but retains animation
    // and plot settings.
    std::filesystem::copy_file(parsfile, dir / "Run_visualizer.par",
                               std::filesystem::copy_options::overwrite_existing);
    // This adapter obtains state via vs_get_var_ptr, not the EXPORT array.
    // Some GUI export sets include driver-model names absent in a direct SDK
    // run; omit those unused channels in this private Parsfile copy.
    const auto staged_parsfile = dir / "Run_all.par";
    {
        std::ifstream pars_input(parsfile);
        std::ofstream pars_output(staged_parsfile, std::ios::binary);
        if (!pars_input || !pars_output)
            throw std::runtime_error("Cannot stage CarSim Parsfile");
        std::string pars_line;
        while (std::getline(pars_input, pars_line)) {
            std::istringstream fields(pars_line);
            std::string keyword;
            fields >> keyword;
            if (keyword == "EXPORT" || keyword == "PORTS_EXP") continue;
            pars_output << pars_line << '\n';
        }
        if (!pars_output) throw std::runtime_error("Cannot finish staged CarSim Parsfile");
    }
    const auto prefix = dir / "LastRun";
    const std::map<std::string, std::filesystem::path> replacements{
        {"FILEBASE", prefix}, {"INPUT", staged_parsfile},
        {"INPUTARCHIVE", dir / "LastRun_all.par"},
        {"ECHO", dir / "LastRun_echo.par"},
        {"FINAL", dir / "LastRun_end.par"},
        {"LOGFILE", dir / "LastRun_log.txt"},
        {"ERDFILE", dir / "LastRun.vs"}};
    const auto staged = dir / "simfile.sim";
    std::ofstream output(staged, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot write staged CarSim Simfile");
    for (const auto& original : lines) {
        std::istringstream fields(original);
        std::string keyword;
        fields >> keyword;
        if (keyword == "PORTS_EXP") {
            output << "PORTS_EXP 0\n";
            continue;
        }
        const auto found = replacements.find(keyword);
        if (found == replacements.end()) output << original << '\n';
        else output << keyword << ' ' << found->second.string() << '\n';
    }
    if (!output) throw std::runtime_error("Cannot finish staged CarSim Simfile");

    // Match the wrapper generated by the CarSim Browser for VS Visualizer:
    // load the native result and the complete animation/plot Parsfile together.
    {
        std::ofstream visualizer(dir / "visualizer.par", std::ios::binary);
        if (!visualizer) throw std::runtime_error("Cannot write CarSim Visualizer Parsfile");
        visualizer << "PARSFILE\nsearch_path_append .\n";
        if (!data_dir.empty())
            visualizer << "search_path_append " << data_dir.string() << '\n';
        if (!program_dir.empty()) {
            visualizer << "search_path_append " << (program_dir / "Resources").string() << '\n'
                       << "search_path_append " << program_dir.string() << '\n';
        }
        visualizer << "set_run_slot 0\n"
                   << "DATASET " << (dir / "LastRun.vs").string() << '\n'
                   << "PARSFILE " << (dir / "Run_visualizer.par").string() << '\n'
                   << "END\n";
        if (!visualizer) throw std::runtime_error("Cannot finish CarSim Visualizer Parsfile");
    }
#ifdef _WIN32
    const auto visualizer_exe = program_dir / "Programs" / "VsVisualizer" / "VsVisualizer.exe";
    if (std::filesystem::is_regular_file(visualizer_exe)) {
        std::ofstream launcher(dir / "open_visualizer.cmd", std::ios::binary);
        if (!launcher) throw std::runtime_error("Cannot write CarSim Visualizer launcher");
        launcher << "@echo off\r\nstart \"\" \"" << visualizer_exe.string()
                 << "\" \"%~dp0visualizer.par\"\r\n";
        if (!launcher) throw std::runtime_error("Cannot finish CarSim Visualizer launcher");
    }
#endif
    return staged;
}

std::filesystem::path syncCarSimBrowserResult(const std::filesystem::path& source_simfile,
                                               const std::filesystem::path& native_dir) {
    std::ifstream input(source_simfile);
    if (!input) throw std::runtime_error("Cannot read source CarSim Simfile for Browser sync");
    std::map<std::string, std::string> macros;
    std::filesystem::path data_dir;
    std::string input_value, filebase_value, line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string keyword, value;
        fields >> keyword;
        if (keyword == "SET_MACRO") {
            std::string name;
            fields >> name;
            std::getline(fields, value);
            macros[name] = trimSimfileValue(value);
        } else if (keyword == "DATADIR" || keyword == "INPUT" || keyword == "FILEBASE") {
            std::getline(fields, value);
            value = trimSimfileValue(value);
            if (keyword == "DATADIR") data_dir = value;
            else if (keyword == "INPUT") input_value = value;
            else filebase_value = value;
        }
    }
    if (input_value.empty() || filebase_value.empty())
        throw std::runtime_error("CarSim Simfile lacks INPUT or FILEBASE for Browser sync");
    const auto origin = std::filesystem::absolute(source_simfile).parent_path();
    if (data_dir.empty()) data_dir = origin;
    else if (data_dir.is_relative()) data_dir = origin / data_dir;
    auto source_par = std::filesystem::path(expandSimfileMacros(input_value, macros));
    auto prefix = std::filesystem::path(expandSimfileMacros(filebase_value, macros));
    if (source_par.is_relative()) source_par = data_dir / source_par;
    if (prefix.is_relative()) prefix = data_dir / prefix;
    source_par = std::filesystem::absolute(source_par).lexically_normal();
    prefix = std::filesystem::absolute(prefix).lexically_normal();
    if (source_par.parent_path() != prefix.parent_path())
        throw std::runtime_error("CarSim FILEBASE is outside the source Run Control result directory");
    const auto archive = std::filesystem::absolute(native_dir).lexically_normal();
    if (!sameFileContents(source_par, archive / "Run_visualizer.par"))
        throw std::runtime_error("CarSim source parameters changed since this run; Browser result was not updated");

    const auto gui_file = [&prefix](const std::string& suffix) {
        return std::filesystem::path(prefix.string() + suffix);
    };
    const std::vector<std::pair<std::filesystem::path, std::filesystem::path>> files{
        {archive / "LastRun.vsb", gui_file(".vsb")},
        {archive / "Run_visualizer.par", gui_file("_all.par")},
        {archive / "LastRun_echo.par", gui_file("_echo.par")},
        {archive / "LastRun_end.par", gui_file("_end.par")},
        {archive / "LastRun_log.txt", gui_file("_log.txt")},
        {archive / "LastRun.vs", gui_file(".vs")}};
    for (const auto& [from, to] : files)
        if (!std::filesystem::is_regular_file(from))
            throw std::runtime_error("CarSim native result missing for Browser sync: " + from.string());
    const auto backup_dir = archive / "browser_previous_cache";
    for (const auto& [from, to] : files) {
        if (std::filesystem::is_regular_file(to)) {
            std::filesystem::create_directories(backup_dir);
            std::filesystem::copy_file(to, backup_dir / to.filename(),
                                       std::filesystem::copy_options::overwrite_existing);
        }
    }
    for (const auto& [from, to] : files)
        std::filesystem::copy_file(from, to,
                                   std::filesystem::copy_options::overwrite_existing);
    return gui_file(".vs");
}

CarSimPlant::CarSimPlant(CarSimConfig config) : config_(std::move(config)) {
#ifndef _WIN32
    throw std::runtime_error("CarSim VS Solver DLL requires Windows");
#else
    if (!std::filesystem::is_regular_file(config_.simfile) ||
        !std::filesystem::is_regular_file(config_.solver_dll))
        throw std::invalid_argument("CarSim requires a generated Simfile and Solver DLL");
    if (config_.simfile.extension() != ".sim")
        throw std::invalid_argument("CarSim input must be a generated .sim file, not a .cpar archive");
    // Catch the common Run Control mistake before loading the licensed solver.
    {
        std::ifstream input(config_.simfile);
        std::string line;
        while (std::getline(input, line)) {
            std::istringstream fields(line);
            std::string keyword;
            int count = -1;
            if (fields >> keyword >> count && keyword == "PORTS_IMP" && count == 0)
                throw std::invalid_argument(
                    "CarSim Simfile has PORTS_IMP 0; configure a steering IMPORT "
                    "with Replace in Run Control and regenerate the Simfile");
        }
    }
    HMODULE dll = LoadLibraryW(std::filesystem::absolute(config_.solver_dll).c_str());
    if (!dll) throw std::runtime_error("Cannot load CarSim Solver DLL");
    library_ = dll;
    try {
        auto required = [dll](const char* name) -> FARPROC {
            const auto proc = GetProcAddress(dll, name);
            if (!proc) throw std::runtime_error(std::string("Missing CarSim VS API: ") + name);
            return proc;
        };
        read_configuration_ = reinterpret_cast<ReadConfiguration>(required("vs_read_configuration"));
        initialize_ = reinterpret_cast<Initialize>(required("vs_initialize"));
        integrate_io_ = reinterpret_cast<IntegrateIo>(required("vs_integrate_io"));
        error_occurred_ = reinterpret_cast<ErrorOccurred>(required("vs_error_occurred"));
        error_message_ = reinterpret_cast<ErrorMessage>(required("vs_get_error_message"));
        get_var_ptr_ = reinterpret_cast<GetVarPtr>(required("vs_get_var_ptr"));
        terminate_run_ = reinterpret_cast<TerminateRun>(required("vs_terminate_run"));
        set_error_dialog_ = reinterpret_cast<SetErrorDialog>(required("vs_set_opt_error_dialog"));
        set_error_dialog_(0);
        int n_import = 0, n_export = 0;
        double stop = 0.0;
        const auto simfile = std::filesystem::absolute(config_.simfile).string();
        read_configuration_(simfile.c_str(), &n_import, &n_export, &time_,
                            &stop, &solver_dt_);
        if (error_occurred_()) throw std::runtime_error(solverError());
        initialize_(time_, nullptr, nullptr);
        initialized_ = true;
        if (error_occurred_()) throw std::runtime_error(solverError());
        if (!(solver_dt_ > 0.0) || n_import < 1 || n_export < 0 ||
            n_import > 4096 || n_export > 4096)
            throw std::runtime_error("Invalid CarSim integration step or I/O channel counts");
        if (config_.steering_left_index < 0 ||
            config_.steering_left_index >= n_import ||
            config_.steering_right_index >= n_import)
            throw std::runtime_error("Steering IMPORT index is outside the CarSim channel array");
        imports_.assign(n_import, 0.0);
        exports_.assign(std::max(1, n_export), 0.0);
        for (std::size_t i = 0; i < state_ptrs_.size(); ++i) {
            std::string keyword = config_.state_keywords[i];
            state_ptrs_[i] = get_var_ptr_(keyword.data());
            if (!state_ptrs_[i])
                throw std::runtime_error("CarSim state variable unavailable: " + keyword);
        }
        (void)initialState();
    } catch (...) {
        if (initialized_) terminate_run_(time_);
        FreeLibrary(dll);
        library_ = nullptr;
        throw;
    }
#endif
}

CarSimPlant::~CarSimPlant() {
#ifdef _WIN32
    if (initialized_) terminate_run_(time_);
    if (library_) FreeLibrary(static_cast<HMODULE>(library_));
#endif
}

std::string CarSimPlant::solverError() const {
    const char* message = error_message_ ? error_message_() : nullptr;
    return message && *message ? message : "CarSim Solver reported an error";
}

VehicleState CarSimPlant::initialState() const {
    VehicleState state;
    state.x = *state_ptrs_[0];
    state.y = *state_ptrs_[1];
    state.psi = *state_ptrs_[2];
    state.vx = *state_ptrs_[3];
    state.vy = *state_ptrs_[4];
    state.r = *state_ptrs_[5];
    for (double value : {state.x, state.y, state.psi, state.vx, state.vy, state.r})
        if (!std::isfinite(value)) throw std::runtime_error("CarSim returned a nonfinite state");
    return state;
}

void CarSimPlant::step(VehicleState& state, double delta_front, double dt) const {
    if (!std::isfinite(delta_front) || !(dt > 0.0))
        throw std::invalid_argument("Invalid CarSim steering command or control step");
    const double ratio = dt / solver_dt_;
    const auto count = static_cast<int>(std::llround(ratio));
    if (count < 1 || std::abs(ratio - count) > 1.0e-6)
        throw std::invalid_argument("MPC dt must be an integer multiple of CarSim solver step");
    const double previous_vx = state.vx;
    std::fill(imports_.begin(), imports_.end(), 0.0);
    const double input = delta_front * config_.steering_gain;
    imports_[config_.steering_left_index] = input;
    if (config_.steering_right_index >= 0)
        imports_[config_.steering_right_index] = input;
    for (int i = 0; i < count; ++i) {
        const int status = integrate_io_(time_, imports_.data(), exports_.data());
        if (error_occurred_()) throw std::runtime_error(solverError());
        if (status) throw std::runtime_error("CarSim ended before the requested control step");
        time_ += solver_dt_;
    }
    state = initialState();
    state.ax = (state.vx - previous_vx) / dt;
}
