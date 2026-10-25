cmake_minimum_required(VERSION 3.16)
project(MPC_DoubleLaneChange LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MPC_CORE
    src/config.cpp
    src/reference.cpp
    src/vehicle.cpp
    src/carsim_plant.cpp
    src/mpc.cpp
    src/rl.cpp
)

add_executable(MPC_DoubleLaneChange
    src/main.cpp
    src/visualization.cpp
    ${MPC_CORE}
)
target_include_directories(MPC_DoubleLaneChange PRIVATE src)
if(WIN32)
    target_link_libraries(MPC_DoubleLaneChange PRIVATE shell32)
endif()
if(MSVC)
    # Keep Debug symbols in object files: avoids MSVC PDB server version conflicts.
    target_compile_options(MPC_DoubleLaneChange PRIVATE /W4 /utf-8 $<$<CONFIG:Debug>:/Z7>)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MPC_DoubleLaneChange)
    set_target_properties(MPC_DoubleLaneChange PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VS_DEBUGGER_COMMAND_ARGUMENTS "--config \"${CMAKE_CURRENT_SOURCE_DIR}/configs/default.ini\"")
else()
    target_compile_options(MPC_DoubleLaneChange PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(MPC_BUILD_TESTS "Build MPC numerical and configuration tests" ON)
if(MPC_BUILD_TESTS)
    enable_testing()
    add_executable(mpc_tests tests/verify.cpp src/visualization.cpp ${MPC_CORE})
    target_include_directories(mpc_tests PRIVATE src)
    target_compile_definitions(mpc_tests PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    if(MSVC)
        target_compile_options(mpc_tests PRIVATE /W4 /utf-8 $<$<CONFIG:Debug>:/Z7>)
    else()
        target_compile_options(mpc_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME mpc_numerical_and_config_tests COMMAND mpc_tests)
endif()
