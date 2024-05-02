set(DLLM_COMPUTE_SRCS "")
add_subdirectory(compute)

set(DLLM_COMMUNICATION_SRCS "")
add_subdirectory(communication)

add_library(dllm SHARED
        logger.cpp
        thread_pool.cpp
        tensor.cpp
        ${DLLM_COMPUTE_SRCS}
        ${DLLM_COMMUNICATION_SRCS}
)

target_include_directories(dllm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")

target_link_libraries(dllm PRIVATE CUDA::cudart CUDA::cublas MPI::MPI_CXX nccl)
target_link_libraries(dllm PRIVATE fmt::fmt-header-only)
target_link_libraries(dllm PUBLIC spdlog::spdlog_header_only)
target_link_libraries(dllm PUBLIC nvidia::cutlass::cutlass)
