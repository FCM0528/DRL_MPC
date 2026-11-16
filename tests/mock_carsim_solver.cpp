// An SDK-boundary fixture, without a CarSim installation or license.
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>

namespace {
int export_count = 0;
// All pointers returned by the solver API are SI, unlike the EXPORT array.
std::array<double, 12> state{};
constexpr const char* keywords[] = {
    "XCG_TM", "YCG_TM", "YAW", "VX", "VY", "AVZ", "STEER_L1", "STEER_R1",
    "AX", "AY", "BETA", "RHO_ROAD"};
}

extern "C" {
__declspec(dllexport) void vs_read_configuration(const char* path, int* imports, int* exports,
                                                double* start, double* stop, double* dt) {
    std::ifstream(path) >> export_count;
    *imports = 3;
    *exports = export_count;
    *start = 0.0;
    *stop = 20.0;
    *dt = 0.001;
    state = {5.0, 10.0, 0.03, 20.0, 0.4, 0.05, 0.1, -0.2, 1.25, -2.5, 0.02, 0.003};
}
__declspec(dllexport) void vs_initialize(double, void (*)(double, int), void (*)(int)) {}
__declspec(dllexport) void vs_copy_export_vars(double* values) {
    // Deliberately unrelated to the wheel angles, as in a reordered Run Control.
    for (int i = 0; i < export_count; ++i) values[i] = 100.0 + i;
}
__declspec(dllexport) int vs_integrate_io(double, double* imports, double* exports) {
    state[0] += imports[2] * 0.001; // verifies uncontrolled import's initial value survives
    state[6] = imports[0] * 3.14159265358979323846 / 180.0;
    state[7] = imports[1] * 3.14159265358979323846 / 180.0;
    vs_copy_export_vars(exports);
    return 0;
}
__declspec(dllexport) int vs_error_occurred() { return 0; }
__declspec(dllexport) char* vs_get_error_message() { static char message[] = ""; return message; }
__declspec(dllexport) double* vs_get_var_ptr(char* keyword) {
    for (std::size_t i = 0; i < state.size(); ++i)
        if (std::strcmp(keyword, keywords[i]) == 0) return &state[i];
    return nullptr;
}
__declspec(dllexport) void vs_terminate_run(double) {}
__declspec(dllexport) void vs_set_opt_error_dialog(int) {}
}
