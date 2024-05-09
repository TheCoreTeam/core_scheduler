#pragma once
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

#include <optional>

#include "tensor.h"

// TODO: may fix this later
#define TORCH_CHECK(statement, ...) \
  do {                              \
    if (!(statement)) {             \
      throw;                        \
    }                               \
  } while (false)

namespace torch {
template <int N>
using Tensor = dllm::Tensor<N>;

using Dtype = dllm::Dtype;

constexpr auto kFloat32 = dllm::Dtype::R_32F;
constexpr auto kFloat16 = dllm::Dtype::R_16F;
constexpr auto kBFloat16 = dllm::Dtype::R_16BF;
constexpr auto kInt64 = dllm::Dtype::R_64I;
constexpr auto kInt32 = dllm::Dtype::R_32I;
constexpr auto kUInt8 = dllm::Dtype::R_8U;
constexpr auto kChar = dllm::Dtype::R_8U;

constexpr auto kCUDA = dllm::DeviceType::CUDA;
}  // namespace torch

namespace c10 {
template <typename T>
using optional = std::optional<T>;
}
