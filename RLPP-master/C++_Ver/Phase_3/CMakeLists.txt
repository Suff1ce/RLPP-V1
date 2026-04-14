cmake_minimum_required(VERSION 3.20)
project(RLPP_Phase_3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(RLPP_Phase_3
    source/rlpp_phase3_main.cpp
    source/rlpp_training_math.cpp
    ../Phase_1/source/csv_utils.cpp
)

target_include_directories(RLPP_Phase_3 PRIVATE
    ${CMAKE_SOURCE_DIR}/../Include
    ${CMAKE_SOURCE_DIR}/../External/eigen
)

add_executable(RLPP_Phase_3_Trainer
    source/rlpp_phase3_trainer_main.cpp
    source/rlpp_phase3_training_case_parity.cpp
    source/rlpp_training_math.cpp
    ../Phase_1/source/csv_utils.cpp
)

target_include_directories(RLPP_Phase_3_Trainer PRIVATE
    ${CMAKE_SOURCE_DIR}/../Include
    ${CMAKE_SOURCE_DIR}/../External/eigen
)

add_executable(RLPP_Phase_3_LoaderEmuVerify
    source/rlpp_phase3_loader_emulator_verify.cpp
    source/rlpp_dataloader.cpp
    source/rlpp_emulator.cpp
    source/rlpp_decoding_params.cpp
    ../Phase_1/source/csv_utils.cpp
    ../Phase_1/source/decoding_model01.cpp
)

target_include_directories(RLPP_Phase_3_LoaderEmuVerify PRIVATE
    ${CMAKE_SOURCE_DIR}/../Include
    ${CMAKE_SOURCE_DIR}/../External/eigen
)

add_executable(RLPP_Phase_3_Run
    source/rlpp_phase3_run_main.cpp
    source/rlpp_phase3_training_case_parity.cpp
    source/rlpp_dataloader.cpp
    source/rlpp_emulator.cpp
    source/rlpp_training_math.cpp
    source/rlpp_decoding_params.cpp
    ../Phase_1/source/csv_utils.cpp
    ../Phase_1/source/decoding_model01.cpp
)

target_include_directories(RLPP_Phase_3_Run PRIVATE
    ${CMAKE_SOURCE_DIR}/../Include
    ${CMAKE_SOURCE_DIR}/../External/eigen
)

add_executable(RLPP_Phase_3_FullLoopVerify
    source/rlpp_phase3_full_loop_emulator_verify.cpp
    source/rlpp_emulator.cpp
    source/rlpp_training_math.cpp
    source/rlpp_decoding_params.cpp
    ../Phase_1/source/csv_utils.cpp
    ../Phase_1/source/decoding_model01.cpp
)

target_include_directories(RLPP_Phase_3_FullLoopVerify PRIVATE
    ${CMAKE_SOURCE_DIR}/../Include
    ${CMAKE_SOURCE_DIR}/../External/eigen
)

add_executable(RLPP_Phase_3_FullLoopTrainedVerify
    source/rlpp_phase3_full_loop_trained_decoder_verify.cpp
    source/rlpp_emulator.cpp
    source/rlpp_training_math.cpp
    source/rlpp_decoding_params.cpp
    ../Phase_1/source/csv_utils.cpp
    ../Phase_1/source/decoding_model01.cpp
)

target_include_directories(RLPP_Phase_3_FullLoopTrainedVerify PRIVATE
    ${CMAKE_SOURCE_DIR}/../Include
    ${CMAKE_SOURCE_DIR}/../External/eigen
)

add_executable(RLPP_Phase_3_RunCaseVerify
    source/rlpp_phase3_run_case_verify.cpp
    source/rlpp_emulator.cpp
    source/rlpp_training_math.cpp
    source/rlpp_decoding_params.cpp
    ../Phase_1/source/csv_utils.cpp
    ../Phase_1/source/decoding_model01.cpp
)

target_include_directories(RLPP_Phase_3_RunCaseVerify PRIVATE
    ${CMAKE_SOURCE_DIR}/../Include
    ${CMAKE_SOURCE_DIR}/../External/eigen
)
