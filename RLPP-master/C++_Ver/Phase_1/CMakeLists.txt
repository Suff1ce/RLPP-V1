cmake_minimum_required(VERSION 3.20)
project(RLPP_Phase_1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(RLPP_Phase_1 
    source/main.cpp
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
)

target_include_directories(RLPP_Phase_1 PRIVATE 
    ${CMAKE_SOURCE_DIR}/../External/eigen 
    ${CMAKE_SOURCE_DIR}/../Include
)