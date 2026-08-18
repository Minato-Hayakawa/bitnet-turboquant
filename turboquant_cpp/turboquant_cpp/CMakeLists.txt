cmake_minimum_required(VERSION 3.14)
project(turboquant_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Header-only library target -- consumers can just:
#   target_link_libraries(your_target PRIVATE turboquant)
add_library(turboquant INTERFACE)
target_include_directories(turboquant INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

option(TURBOQUANT_BUILD_TESTS "Build tests" ON)
option(TURBOQUANT_BUILD_EXAMPLES "Build examples" ON)

if(TURBOQUANT_BUILD_TESTS)
    enable_testing()
    add_executable(test_turboquant tests/test_turboquant.cpp)
    target_link_libraries(test_turboquant PRIVATE turboquant)
    add_test(NAME turboquant_tests COMMAND test_turboquant)
endif()

if(TURBOQUANT_BUILD_EXAMPLES)
    add_executable(tq_benchmark examples/benchmark.cpp)
    target_link_libraries(tq_benchmark PRIVATE turboquant)
endif()
