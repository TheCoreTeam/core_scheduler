#pragma once
#include <spdlog/spdlog.h>

namespace dllm {
#define CHECK_CUDART(statement)                                             \
  do {                                                                      \
    auto code = statement;                                                  \
    if (code != cudaSuccess) {                                              \
      SPDLOG_LOGGER_CRITICAL(&::dllm::logger(), "cuda Failed with code {}", \
                             static_cast<int>(code));                       \
    }                                                                       \
  } while (0)

#define CHECK_CUBLAS(statement)                                               \
  do {                                                                        \
    auto code = statement;                                                    \
    if (code != CUBLAS_STATUS_SUCCESS) {                                      \
      SPDLOG_LOGGER_CRITICAL(&::dllm::logger(), "cuBlas Failed with code {}", \
                             static_cast<int>(code));                         \
    }                                                                         \
  } while (0)

spdlog::logger &logger();
}  // namespace dllm
