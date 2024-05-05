set(DLLM_COMPUTE_SRCS "")
add_subdirectory(compute)

set(DLLM_COMMUNICATION_SRCS "")
add_subdirectory(communication)

set(DLLM_OPTIMIZER_SRCS "")
add_subdirectory(optimizer)

set(DLLM_THREADING_SRCS "")
add_subdirectory(threading)

add_library(dllm_core SHARED
        logger.cpp
        tensor.cpp
        ${DLLM_COMPUTE_SRCS}
        ${DLLM_COMMUNICATION_SRCS}
        ${DLLM_OPTIMIZER_SRCS}
        ${DLLM_THREADING_SRCS}
)

target_compile_options(dllm_core PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
)

target_include_directories(dllm_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")

target_link_libraries(dllm_core PRIVATE CUDA::cudart CUDA::cublas MPI::MPI_CXX nccl)
target_link_libraries(dllm_core PRIVATE fmt::fmt-header-only)
target_link_libraries(dllm_core PUBLIC spdlog::spdlog_header_only)
target_link_libraries(dllm_core PUBLIC nvidia::cutlass::cutlass)

add_subdirectory(flash_attention)

add_library(dllm INTERFACE)

target_link_libraries(dllm INTERFACE dllm_core dllm_flash_attention_plugin)
