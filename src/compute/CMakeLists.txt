set(DLLM_COMPUTE_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/mse.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/mse.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/relu.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/relu.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/init.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/init.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/fc.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp"
)
set(DLLM_COMPUTE_SRCS ${DLLM_COMPUTE_SRCS} PARENT_SCOPE)
