set(DLLM_COMPUTE_SRCS "")
add_subdirectory(compute)

set(DLLM_MEMORY_SRCS "")
add_subdirectory(memory)

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
        ${DLLM_MEMORY_SRCS}
        ${DLLM_COMMUNICATION_SRCS}
        ${DLLM_OPTIMIZER_SRCS}
        ${DLLM_THREADING_SRCS}
)

target_compile_options(dllm_core PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--expt-relaxed-constexpr;--extended-lambda;-O3;-Wall;-Xcompiler;-Ofast;-Wall>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;--expt-relaxed-constexpr;--extended-lambda;-O0;-Wall;-Xcompiler;-Ofast;-Wall>
        $<$<AND:$<CONFIG:Release>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-Ofast;-Wall>
        $<$<AND:$<CONFIG:Debug>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-O0;-Wall>
)

target_include_directories(dllm_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(dllm_core PRIVATE CUDA::cudart_static CUDA::cublas CUDA::nvtx3)
target_link_libraries(dllm_core PRIVATE fmt::fmt-header-only)
target_link_libraries(dllm_core PUBLIC spdlog::spdlog_header_only OpenMP::OpenMP_CXX nvidia::nccl MPI::MPI_C torch)

target_compile_definitions(dllm_core PRIVATE -DDLLM_ENABLE_INTERNAL_BUILD)

add_library(dllm INTERFACE)

target_link_libraries(dllm INTERFACE dllm_core)
