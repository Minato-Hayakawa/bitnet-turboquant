# TurboQuant is header-only.  Keeping this target separate means it can be
# built and tested before the llama.cpp submodule is available.
add_library(bitnet_turboquant_kv INTERFACE)
target_include_directories(bitnet_turboquant_kv INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../turboquant_cpp/turboquant_cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(bitnet_turboquant_kv INTERFACE cxx_std_17)

add_executable(bitnet-turboquant-eval src/bitnet_turboquant_eval.cpp)
target_link_libraries(bitnet-turboquant-eval PRIVATE bitnet_turboquant_kv)
