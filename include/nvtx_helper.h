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
#include <nvtx3/nvToolsExt.h>

namespace cs {
struct NvtxRange {
  template <typename... T>
  explicit NvtxRange(const std::string &range_string) {
    nvtxRangePush(range_string.c_str());
  }
  ~NvtxRange() { nvtxRangePop(); }
};

// Helper macros to convert macro values to strings
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// Helper macro to concatenate two tokens
#define CONCAT(a, b) a##b

// Macro to create a unique variable name based on __LINE__
#define UNIQUE_VAR(prefix) CONCAT(prefix##_at_##__LINE__, _unique)

// Define the NVTX range macro using the unique variable name
#define CS_NVTX_RANGE_FN(FUNC_NAME) \
  ::cs::NvtxRange UNIQUE_VAR(__nvtx_range)(FUNC_NAME)
}  // namespace cs
