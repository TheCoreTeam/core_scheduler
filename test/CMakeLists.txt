add_executable(dllm_tests
        main.cpp
        test_adamw.cpp
        test_linear.cpp
        test_gelu.cpp
        test_flash_attention.cpp
        test_embedding.cpp
        test_cross_entropy.cpp
        test_utils.cpp
        test_layer_norm.cpp
)

target_link_libraries(dllm_tests PRIVATE
        dllm
        GTest::gtest
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
        test_reduce_scatter.cpp
)

target_include_directories(dllm_communication_tests PRIVATE
        ${TORCH_INCLUDE_PATH}
)

target_link_libraries(dllm_communication_tests PRIVATE
        dllm
        GTest::gtest
)

target_compile_options(dllm_communication_tests PRIVATE
        $<$<CONFIG:Release>:-Ofast>
        $<$<CONFIG:Debug>:-O0>
)
