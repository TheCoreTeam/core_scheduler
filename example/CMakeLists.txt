function(create_executable target_name source_files)
    add_executable(${target_name} ${source_files})
    target_link_libraries(${target_name} PRIVATE
            dllm
            spdlog::spdlog
            CUDA::cudart
            CUDA::cublas
            Eigen3::Eigen
            MPI::MPI_CXX
            nccl
    )
    target_compile_options(${target_name} PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Debug>:-O0>
    )
endfunction()

create_executable(dllm_fc_regression fc_regression.cpp)
create_executable(dllm_distributed_fc_regression distributed_fc_regression.cpp)
