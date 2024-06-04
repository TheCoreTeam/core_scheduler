set(DLLM_COMPUTE_SRCS
#        ${CMAKE_CURRENT_SOURCE_DIR}/relu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cross_entropy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gelu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/add.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/embedding.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scaled_dot_product_attention.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.cpp
)
set(DLLM_COMPUTE_SRCS ${DLLM_COMPUTE_SRCS} PARENT_SCOPE)
