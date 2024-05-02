add_library(dllm SHARED
        logger.cpp
        thread_pool.cpp
        tensor.cpp
        matmul.cpp
        fc.cpp
)

target_include_directories(dllm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")

target_link_libraries(dllm PRIVATE CUDA::cudart CUDA::cublas MPI::MPI_CXX)
target_link_libraries(dllm PRIVATE fmt::fmt-header-only)
target_link_libraries(dllm PUBLIC spdlog::spdlog)
target_link_libraries(dllm PUBLIC nvidia::cutlass::cutlass)
