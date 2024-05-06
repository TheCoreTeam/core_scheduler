add_executable(dllm_tests
        main.cpp
        test_fc.cpp
)

target_link_libraries(dllm_tests PRIVATE
        dllm
        spdlog::spdlog
        CUDA::cudart
        CUDA::cublas
        Eigen3::Eigen
        GTest::gtest
)

target_compile_options(dllm_tests PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
)

add_executable(dllm_communication_tests
        main_communication.cpp
        test_all_reduce.cpp
)

target_link_libraries(dllm_communication_tests PRIVATE
        dllm
        spdlog::spdlog
        CUDA::cudart
        CUDA::cublas
        Eigen3::Eigen
        GTest::gtest
        MPI::MPI_CXX
        ${NCCL_LIBRARIES}
)

target_compile_options(dllm_communication_tests PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
)
