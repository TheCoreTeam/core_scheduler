/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <string>

namespace cs {
#define CS_ASSERT_TRUE(statement, ...)                                      \
  do {                                                                        \
    if (!(statement)) {                                                       \
      std::string errorMessage =                                              \
          fmt::format("Assert '{}' failed: '{}'", #statement, ##__VA_ARGS__); \
      SPDLOG_LOGGER_CRITICAL(&::cs::logger(), errorMessage);                \
      std::abort();                                                           \
    }                                                                         \
  } while (false)

#define CS_WARN_TRUE(statement, ...)                                        \
  do {                                                                        \
    if (!(statement)) {                                                       \
      std::string errorMessage =                                              \
          fmt::format("Assert '{}' failed: '{}'", #statement, ##__VA_ARGS__); \
      SPDLOG_LOGGER_WARN(&::cs::logger(), errorMessage);                    \
    }                                                                         \
  } while (false)

#define CHECK_CUDART(statement)                                        \
  do {                                                                 \
    auto code = (statement);                                           \
    CS_ASSERT_TRUE((code) == cudaSuccess, cudaGetErrorString(code)); \
  } while (false)

#define CHECK_CUBLAS(statement)                                               \
  do {                                                                        \
    auto code = (statement);                                                  \
    CS_ASSERT_TRUE((code) == CUBLAS_STATUS_SUCCESS,                         \
                     fmt::format("statement {} returned code {}", #statement, \
                                 static_cast<int>(code)));                    \
  } while (false)

#define CHECK_MPI(statement)                                                  \
  do {                                                                        \
    auto code = (statement);                                                  \
    CS_ASSERT_TRUE((code) == MPI_SUCCESS,                                   \
                     fmt::format("statement {} returned code {}", #statement, \
                                 static_cast<int>(code)));                    \
  } while (false)

#define CHECK_NCCL(statement)                                                 \
  do {                                                                        \
    auto code = (statement);                                                  \
    CS_ASSERT_TRUE((code) == ncclSuccess,                                   \
                     fmt::format("statement {} returned code {}", #statement, \
                                 static_cast<int>(code)));                    \
  } while (false)

spdlog::logger &logger();
}  // namespace cs
