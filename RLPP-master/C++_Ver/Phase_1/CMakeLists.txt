cmake_minimum_required(VERSION 3.20)
project(RLPP_Phase_1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(RLPP_Phase_1 
    source/main.cpp
    source/rlpp_phase1_cli.cpp
    source/csv_utils.cpp
    source/math_functions.cpp
    source/two_layer_mlp.cpp
    source/exponential_history_encoder.cpp
    source/decoder_history_buffer.cpp
    source/F1_rat01_bundle.cpp
    source/F1_rat01_validate.cpp
    source/F1_rat01_parity.cpp
    source/decoding_model01.cpp
    source/replay_runner.cpp
    source/RLPP_inference.cpp
    source/rlpp_hardware_output.cpp
)

target_include_directories(RLPP_Phase_1 PRIVATE 
    ${CMAKE_SOURCE_DIR}/../External/eigen 
    ${CMAKE_SOURCE_DIR}/../Include
)

# UDP loopback: transport + device-side v1 unpack (no Eigen)
add_executable(RLPP_hardware_udp_loopback
    source/hardware_frame_v1.cpp
    source/hardware_udp_loopback_main.cpp
)
target_include_directories(RLPP_hardware_udp_loopback PRIVATE ${CMAKE_SOURCE_DIR}/../Include)
if(WIN32)
    target_link_libraries(RLPP_hardware_udp_loopback PRIVATE ws2_32)
endif()
set_target_properties(RLPP_hardware_udp_loopback PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

# Real-time UDP inference scaffold (DAQ thread + model thread + SPSC queue)
add_executable(RLPP_realtime_udp_infer
    source/rlpp_realtime_udp_infer_main.cpp
    source/upstream_frame_v1.cpp
    source/csv_utils.cpp
    source/math_functions.cpp
    source/two_layer_mlp.cpp
    source/exponential_history_encoder.cpp
    source/decoder_history_buffer.cpp
    source/F1_rat01_bundle.cpp
    source/decoding_model01.cpp
    source/RLPP_inference.cpp
    source/rlpp_hardware_output.cpp
)
target_include_directories(RLPP_realtime_udp_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/../External/eigen
    ${CMAKE_SOURCE_DIR}/../Include
)
if(WIN32)
    target_link_libraries(RLPP_realtime_udp_infer PRIVATE ws2_32)
endif()