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
    add_executable(mpc_tests tests/verify.cpp)
    target_link_libraries(mpc_tests PRIVATE mpc_core)
    target_compile_definitions(mpc_tests PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    mpc_compile_options(mpc_tests)
    add_test(NAME mpc_numerical_and_config_tests COMMAND mpc_tests)
endif()
