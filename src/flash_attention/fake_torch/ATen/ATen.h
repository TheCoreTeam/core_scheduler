#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "logger.h"
#include "tensor.h"

#define AT_ERROR(...) SPDLOG_LOGGER_CRITICAL(&::dllm::logger(), __VA_ARGS__);

__inline__ __attribute__((always_inline)) const char* toString(
    dllm::Dtype dtype) {
  switch (dtype) {
    case dllm::R_64F:
      return "FP64";
    case dllm::R_32F:
      return "FP32";
    case dllm::R_16F:
      return "FP16";
    case dllm::R_16BF:
      return "BF16";
    default:
      return nullptr;
  }
}

namespace torch {
template <int N>
using Tensor = dllm::Tensor<N>;
}

namespace at {
using Half = nv_half;
using BFloat16 = nv_bfloat16;
namespace ScalarType {
constexpr auto Half = dllm::R_16F;
constexpr auto BFloat16 = dllm::R_16BF;
}  // namespace ScalarType
}  // namespace at
