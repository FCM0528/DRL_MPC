cmake_minimum_required(VERSION 3.16)
project(MPC_DoubleLaneChange LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(mpc_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /utf-8 $<$<CONFIG:Debug>:/Z7>)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(mpc_core STATIC
    src/config.cpp
    src/reference.cpp
    src/vehicle.cpp
    src/carsim_plant.cpp
    src/mpc.cpp
    src/rl.cpp
    src/visualization.cpp
)
target_include_directories(mpc_core PUBLIC src)
mpc_compile_options(mpc_core)

add_executable(MPC_DoubleLaneChange src/main.cpp)
target_link_libraries(MPC_DoubleLaneChange PRIVATE mpc_core)
mpc_compile_options(MPC_DoubleLaneChange)
if(WIN32)
    target_link_libraries(MPC_DoubleLaneChange PRIVATE shell32)
endif()
if(MSVC)
    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT MPC_DoubleLaneChange)
    set_target_properties(MPC_DoubleLaneChange PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VS_DEBUGGER_COMMAND_ARGUMENTS "--config \"${CMAKE_CURRENT_SOURCE_DIR}/configs/default.ini\"")
endif()

add_executable(MPC_RL_Bridge src/rl_bridge_main.cpp)
target_link_libraries(MPC_RL_Bridge PRIVATE mpc_core)
mpc_compile_options(MPC_RL_Bridge)

option(MPC_BUILD_TESTS "Build MPC numerical and configuration tests" ON)
if(MPC_BUILD_TESTS)
    enable_testing()
    # Retired INIs are reconstructed as build-only regression fixtures, not
    # duplicated in the source tree. Requires the documented recovery commit.
    find_package(Git REQUIRED)
    set(LEGACY_FIXTURES "${CMAKE_CURRENT_BINARY_DIR}/legacy-configs")
    file(MAKE_DIRECTORY "${LEGACY_FIXTURES}")
    foreach(name default.ini mpc_strong.ini strong_actions2.ini strong_actions3.ini strong_actions4.ini strong_actions3_wy2.ini strong_actions3_wy4.ini carsim_python_rl_joint.ini strong_qy_only.ini strong_qy_qpsi.ini carsim_python_rl.ini magic_tire.csv)
        execute_process(COMMAND "${GIT_EXECUTABLE}" show "5f2ece1:configs/${name}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_FILE "${LEGACY_FIXTURES}/${name}" RESULT_VARIABLE restored)
        if(NOT restored EQUAL 0)
            message(FATAL_ERROR "Missing recovery commit 5f2ece1; fetch full repository history before building tests")
        endif()
    endforeach()
    add_executable(mpc_tests tests/verify.cpp)
    target_link_libraries(mpc_tests PRIVATE mpc_core)
    target_compile_definitions(mpc_tests PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(mpc_tests PRIVATE TEST_LEGACY_CONFIG_DIR="${LEGACY_FIXTURES}")
    mpc_compile_options(mpc_tests)
    add_test(NAME mpc_numerical_and_config_tests COMMAND mpc_tests)
    if(WIN32)
        add_library(mock_carsim_solver SHARED tests/mock_carsim_solver.cpp)
        mpc_compile_options(mock_carsim_solver)
        add_dependencies(mpc_tests mock_carsim_solver)
        target_compile_definitions(mpc_tests PRIVATE
            TEST_CARSIM_DLL="$<TARGET_FILE:mock_carsim_solver>")
    endif()
endif()
