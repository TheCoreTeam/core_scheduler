#pragma once
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <string>

namespace dllm {
#define DLLM_ASSERT_TRUE(statement, ...)                                      \
  do {                                                                        \
    if (!(statement)) {                                                       \
      std::string errorMessage =                                              \
          fmt::format("Assert '{}' failed: '{}'", #statement, ##__VA_ARGS__); \
      SPDLOG_LOGGER_CRITICAL(&::dllm::logger(), errorMessage);                \
      throw std::runtime_error(errorMessage);                                 \
    }                                                                         \
  } while (0)

#define CHECK_CUDART(statement) \
  DLLM_ASSERT_TRUE((statement) == cudaSuccess, "No further message")

#define CHECK_CUBLAS(statement) \
  DLLM_ASSERT_TRUE((statement) == CUBLAS_STATUS_SUCCESS, "No further message")

#define CHECK_MPI(statement) \
  DLLM_ASSERT_TRUE((statement) == MPI_SUCCESS, "No further message")

#define CHECK_NCCL(statement) \
  DLLM_ASSERT_TRUE((statement) == ncclSuccess, "No further message")

spdlog::logger &logger();
}  // namespace dllm
