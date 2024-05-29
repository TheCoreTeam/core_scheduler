#pragma once
#include <fmt/format.h>
#include <nvtx3/nvToolsExt.h>

namespace dllm {
struct NvtxRange {
  template <typename... T>
  explicit NvtxRange(fmt::format_string<T...> fmt, T &&...args) {
    const std::string range_string = fmt::format(fmt, std::forward<T>(args)...);
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
#define DLLM_NVTX_RANGE_FN(FUNC_NAME) \
  ::dllm::NvtxRange UNIQUE_VAR(__nvtx_range)(FUNC_NAME)
}  // namespace dllm
