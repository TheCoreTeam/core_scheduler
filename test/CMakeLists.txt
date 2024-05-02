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
)

target_compile_options(dllm_tests PRIVATE
        $<$<CONFIG:Debug>:-O0>
)