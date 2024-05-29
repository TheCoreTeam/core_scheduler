add_executable(dllm_tests
        main.cpp
        test_adamw.cpp
        #        test_relu.cpp
        test_linear.cpp
        test_gelu.cpp
        test_init.cpp
        test_flash_attention.cpp
#        test_add.cpp
        test_embedding.cpp
)

target_link_libraries(dllm_tests PRIVATE
        dllm
        spdlog::spdlog_header_only
        CUDA::cudart_static
        CUDA::cublas
        GTest::gtest
        torch
)

target_compile_options(dllm_tests PRIVATE
        $<$<CONFIG:Release>:-Ofast>
        $<$<CONFIG:Debug>:-O0>
)

add_executable(dllm_communication_tests
        main_communication.cpp
        test_all_reduce.cpp
        test_all_gather.cpp
        test_all_to_all.cpp
)

target_include_directories(dllm_communication_tests PRIVATE
        ${TORCH_INCLUDE_PATH}
)

target_link_libraries(dllm_communication_tests PRIVATE
        dllm
        spdlog::spdlog_header_only
        CUDA::cudart_static
        CUDA::cublas
        GTest::gtest
        MPI::MPI_CXX
        nvidia::nccl
        torch
)

target_compile_options(dllm_communication_tests PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
)
