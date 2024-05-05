set(FLASH_ATTENTION_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/flash-attention")
set(FLASH_ATTENTION_FUSED_SOFTMAX_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/fused_softmax/scaled_masked_softmax_cuda.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/fused_softmax/scaled_upper_triang_masked_softmax_cuda.cu"
)
set(FLASH_ATTENTION_FUSED_SOFTMAX_INCLUDE
        "${FLASH_ATTENTION_PATH}/csrc/fused_softmax/"
)

add_library(dllm_flash_attention_plugin
        ${FLASH_ATTENTION_FUSED_SOFTMAX_SRC}
)

target_include_directories(dllm_flash_attention_plugin PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/fake_torch"
        ${FLASH_ATTENTION_FUSED_SOFTMAX_INCLUDE}
)

target_link_libraries(dllm_flash_attention_plugin PRIVATE dllm CUDA::cudart)

target_compile_options(dllm_flash_attention_plugin PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>
)
