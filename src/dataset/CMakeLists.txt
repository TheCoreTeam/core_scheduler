set(DLLM_DATASET_SRC
        dataset.cpp
        dataloader.cpp
)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
add_library(dllm_dataset SHARED ${DLLM_DATASET_SRC})
target_link_libraries(dllm_dataset PRIVATE dllm_core fmt::fmt-header-only Arrow::arrow_shared Parquet::parquet_shared)
target_link_libraries(dllm_dataset PUBLIC spdlog::spdlog_header_only)

target_compile_options(dllm_dataset PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
)
