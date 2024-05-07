#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "logger.h"
#include "tensor.h"

#define AT_ERROR(...) SPDLOG_LOGGER_CRITICAL(&::dllm::logger(), __VA_ARGS__);

__inline__ __attribute__((always_inline)) const char *toString(
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

namespace dllm {
template <typename T, std::size_t N, std::size_t... I>
__inline__ __attribute__((always_inline)) auto make_shape_impl(
    const std::array<T, N> &array, std::index_sequence<I...>) {
  return cute::make_shape(static_cast<TensorIndexType>(array[I])...);
}

template <typename T, std::size_t N>
__inline__ __attribute__((always_inline)) auto make_shape(
    const std::array<T, N> &array) {
  return make_shape_impl(array, std::make_index_sequence<N>{});
}
}  // namespace dllm

namespace torch {
template <int N>
using Tensor = dllm::Tensor<N>;

// Row Major
template <dllm::DeviceType deviceType, typename IndexType, std::size_t N,
          typename... Args>
__inline__ __attribute__((always_inline)) std::shared_ptr<Tensor<N>> empty(
    const std::array<IndexType, N> &shape, dllm::Dtype dtype, Args &&...args) {
  auto layout = cute::make_layout(dllm::make_shape(shape), cute::GenRowMajor{});
  return Tensor<N>::template empty<deviceType>(layout, dtype,
                                               std::forward<Args>(args)...);
}

template <dllm::DeviceType deviceType, typename Shape, typename Stride,
          typename... Args>
__inline__ __attribute__((always_inline)) auto empty(
    const cute::Layout<Shape, Stride> &layout, dllm::Dtype dtype,
    Args &&...args) {
  constexpr auto N = decltype(rank(layout))::value;
  return Tensor<N>::template empty<deviceType>(layout, dtype,
                                               std::forward<Args>(args)...);
}

// Row Major
template <dllm::DeviceType deviceType, int N, typename... Args>
__inline__ __attribute__((always_inline)) std::shared_ptr<Tensor<N>> empty_like(
    const Tensor<N> &tensor, Args &&...args) {
  return Tensor<N>::template empty<deviceType>(tensor.layout, tensor.dtype,
                                               std::forward<Args>(args)...);
}
template <dllm::DeviceType deviceType, int N, typename... Args>
__inline__ __attribute__((always_inline)) std::shared_ptr<Tensor<N>> empty_like(
    const std::shared_ptr<Tensor<N>> &tensor, Args &&...args) {
  return empty_like<deviceType>(*tensor, std::forward<Args>(args)...);
}
}  // namespace torch

namespace at {
template <int N>
using Tensor = torch::Tensor<N>;
using Half = nv_half;
using BFloat16 = nv_bfloat16;
namespace ScalarType {
constexpr auto Float = dllm::R_32F;
constexpr auto Half = dllm::R_16F;
constexpr auto BFloat16 = dllm::R_16BF;
}  // namespace ScalarType
}  // namespace at
