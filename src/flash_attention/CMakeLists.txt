set(FLASH_ATTENTION_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/flash-attention)
# For debuging:
#set(FLASH_ATTENTION_FLASH_ATTN_SRC
#        ${FLASH_ATTENTION_PATH}/csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu
#        ${FLASH_ATTENTION_PATH}/csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu
#        ${FLASH_ATTENTION_PATH}/csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu
#)
file(GLOB FLASH_ATTENTION_FLASH_ATTN_SRC ${FLASH_ATTENTION_PATH}/csrc/flash_attn/src/*.cu)
list(APPEND FLASH_ATTENTION_FLASH_ATTN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn/flash_api.cpp)
set(FLASH_ATTENTION_FLASH_ATTN_INCLUDE
        ${FLASH_ATTENTION_PATH}/csrc/flash_attn/src/
)

set(FLASH_ATTENTION_FT_ATTEHTION_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/ft_attention/ft_attention.cpp
        ${FLASH_ATTENTION_PATH}/csrc/ft_attention/decoder_masked_multihead_attention.cu
)
set(FLASH_ATTENTION_FT_ATTEHTION_INCLUDE
        ${FLASH_ATTENTION_PATH}/csrc/ft_attention/
)

set(FLASH_ATTENTION_FUSED_SOFTMAX_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_softmax/scaled_masked_softmax_cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_softmax/scaled_upper_triang_masked_softmax_cuda.cu
)
set(FLASH_ATTENTION_FUSED_SOFTMAX_INCLUDE
        ${FLASH_ATTENTION_PATH}/csrc/fused_softmax/
)

# For debuging:
#set(FLASH_ATTENTION_LAYER_NORM_SRC
#        ${FLASH_ATTENTION_PATH}/csrc/layer_norm/ln_bwd_256.cu
#        ${FLASH_ATTENTION_PATH}/csrc/layer_norm/ln_fwd_256.cu
#        ${FLASH_ATTENTION_PATH}/csrc/layer_norm/ln_parallel_bwd_256.cu
#        ${FLASH_ATTENTION_PATH}/csrc/layer_norm/ln_parallel_fwd_256.cu
#)
file(GLOB FLASH_ATTENTION_LAYER_NORM_SRC ${FLASH_ATTENTION_PATH}/csrc/layer_norm/*.cu)
list(APPEND FLASH_ATTENTION_LAYER_NORM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm/ln_api.cpp)
set(FLASH_ATTENTION_LAYER_NORM_INCLUDE
        ${FLASH_ATTENTION_PATH}/csrc/layer_norm/
)

add_library(dll_flash_attention_registry SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm/registry.cpp
)
target_include_directories(dll_flash_attention_registry PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/fake_torch
        ${FLASH_ATTENTION_LAYER_NORM_INCLUDE}
)
target_link_libraries(dll_flash_attention_registry PRIVATE CUDA::cudart)
target_compile_options(dll_flash_attention_registry PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
)

add_library(dllm_flash_attention_plugin SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
        ${FLASH_ATTENTION_FLASH_ATTN_SRC}
        ${FLASH_ATTENTION_FT_ATTEHTION_SRC}
        ${FLASH_ATTENTION_FUSED_SOFTMAX_SRC}
        ${FLASH_ATTENTION_LAYER_NORM_SRC}
)

target_include_directories(dllm_flash_attention_plugin PRIVATE
        ${FLASH_ATTENTION_PATH}/csrc/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/fake_torch
        ${FLASH_ATTENTION_FLASH_ATTN_INCLUDE}
        ${FLASH_ATTENTION_FT_ATTEHTION_INCLUDE}
        ${FLASH_ATTENTION_FUSED_SOFTMAX_INCLUDE}
        ${FLASH_ATTENTION_LAYER_NORM_INCLUDE}
)

target_link_libraries(dllm_flash_attention_plugin PRIVATE dllm CUDA::cudart dll_flash_attention_registry)

target_compile_options(dllm_flash_attention_plugin PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler -Wno-deprecated-enum-enum-conversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>
)

target_compile_definitions(dllm_flash_attention_plugin PRIVATE -DENABLE_BF16 -DDLLM_BUILD_FLASH_ATTENTION)
