add_executable(dllm_fc_regression
        fc_regression.cpp
)

target_link_libraries(dllm_fc_regression PRIVATE
        dllm
        spdlog::spdlog
        CUDA::cudart
        CUDA::cublas
        Eigen3::Eigen
        MPI::MPI_CXX
)

target_compile_options(dllm_fc_regression PRIVATE
        $<$<CONFIG:Release>:-O3>
)

target_compile_options(dllm_fc_regression PRIVATE
        $<$<CONFIG:Debug>:-O0>
)
