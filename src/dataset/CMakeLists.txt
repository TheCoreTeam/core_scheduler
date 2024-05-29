set(DLLM_DATASET_SRC
        dataset.cpp
        dataloader.cpp
)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
add_library(dllm_dataset SHARED ${DLLM_DATASET_SRC})
target_link_libraries(dllm_dataset PRIVATE dllm_core fmt::fmt-header-only arrow_dataset Arrow::arrow_shared Parquet::parquet_shared nvidia::cutlass::cutlass)
target_link_libraries(dllm_dataset PUBLIC spdlog::spdlog_header_only)
target_include_directories(dllm_dataset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/pcg_cpp/include)

target_compile_options(dllm_dataset PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-Ofast;-Wall>
        $<$<AND:$<CONFIG:Debug>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-O0;-Wall>
)
