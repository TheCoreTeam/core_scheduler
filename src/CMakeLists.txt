set(DLLM_ALLOCATE_SRCS "")
add_subdirectory(allocate)

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
        ${DLLM_ALLOCATE_SRCS}
        ${DLLM_COMPUTE_SRCS}
        ${DLLM_COMMUNICATION_SRCS}
        ${DLLM_OPTIMIZER_SRCS}
        ${DLLM_THREADING_SRCS}
)

target_compile_options(dllm_core PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_include_directories(dllm_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${NCCL_INCLUDE_DIRS}
)

target_link_libraries(dllm_core PRIVATE CUDA::cudart CUDA::cublas MPI::MPI_CXX ${NCCL_LIBRARIES})
target_link_libraries(dllm_core PRIVATE fmt::fmt-header-only)
target_link_libraries(dllm_core PUBLIC spdlog::spdlog_header_only)
target_link_libraries(dllm_core PRIVATE nvidia::cutlass::cutlass)

if (DLLM_ENABLE_FLASH_ATTENTION)
    add_subdirectory(flash_attention)
else ()
    add_library(dllm_flash_attention_plugin INTERFACE)
endif ()

add_library(dllm INTERFACE)

target_link_libraries(dllm INTERFACE dllm_core dllm_flash_attention_plugin)
target_link_libraries(dllm INTERFACE nvidia::cutlass::cutlass)
