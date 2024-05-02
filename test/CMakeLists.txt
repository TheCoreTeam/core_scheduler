add_executable(dllm_tests
        main.cpp
        test_fc.cpp
)
target_link_libraries(dllm_tests dllm spdlog::spdlog CUDA::cudart CUDA::cublas Eigen3::Eigen GTest::gtest)
