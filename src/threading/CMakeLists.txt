set(DLLM_THREADING_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_scheduler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/thread_pool_compute.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/thread_pool_cudart.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/thread_stream_cudart.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/thread_stream_nccl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/thread_stream_mpi.cpp
)
set(DLLM_THREADING_SRCS ${DLLM_THREADING_SRCS} PARENT_SCOPE)
