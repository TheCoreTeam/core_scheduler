#pragma once
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <string>

namespace dllm {
#define DLLM_ASSERT_TRUE(statement, ...)                                      \
  do {                                                                        \
    if (!(statement)) {                                                       \
      std::string errorMessage =                                              \
          fmt::format("Assert '{}' failed: '{}'", #statement, ##__VA_ARGS__); \
      SPDLOG_LOGGER_CRITICAL(&::dllm::logger(), errorMessage);                \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

#define DLLM_WARN_TRUE(statement, ...)                                        \
  do {                                                                        \
    if (!(statement)) {                                                       \
      std::string errorMessage =                                              \
          fmt::format("Assert '{}' failed: '{}'", #statement, ##__VA_ARGS__); \
      SPDLOG_LOGGER_WARN(&::dllm::logger(), errorMessage);                    \
    }                                                                         \
  } while (0)

#define CHECK_CUDART(statement)                                        \
  do {                                                                 \
    auto code = (statement);                                           \
    DLLM_ASSERT_TRUE((code) == cudaSuccess, cudaGetErrorString(code)); \
  } while (0)

#define CHECK_CUBLAS(statement)                                               \
  do {                                                                        \
    auto code = (statement);                                                  \
    DLLM_ASSERT_TRUE((code) == CUBLAS_STATUS_SUCCESS,                         \
                     fmt::format("statement {} returned code {}", #statement, \
                                 static_cast<int>(code)));                    \
  } while (0)

#define CHECK_MPI(statement)                                                  \
  do {                                                                        \
    auto code = (statement);                                                  \
    DLLM_ASSERT_TRUE((code) == MPI_SUCCESS,                                   \
                     fmt::format("statement {} returned code {}", #statement, \
                                 static_cast<int>(code)));                    \
  } while (0)

#define CHECK_NCCL(statement)                                                 \
  do {                                                                        \
    auto code = (statement);                                                  \
    DLLM_ASSERT_TRUE((code) == ncclSuccess,                                   \
                     fmt::format("statement {} returned code {}", #statement, \
                                 static_cast<int>(code)));                    \
  } while (0)

spdlog::logger &logger();
}  // namespace dllm
