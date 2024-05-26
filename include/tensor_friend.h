#pragma once
// This is an internal file, never use it unless you know what you are doing
#include "tensor.h"

namespace dllm {
struct TensorFriend {
  template <int N>
  static void resetTensorData(Tensor<N> &tensor,
                              typename Tensor<N>::DataPtr ptr) {
    *tensor.data_ = std::move(ptr);
  }

  template <int N>
  static void resetTensorData(const std::shared_ptr<Tensor<N>> &tensor,
                              typename Tensor<N>::DataPtr ptr) {
    resetTensorData(*tensor, std::move(ptr));
  }

  template <int N>
  static auto getTensorDataPtr(const Tensor<N> &tensor) {
    return tensor.data_;
  }

  template <int N>
  static auto getTensorDataPtr(const std::shared_ptr<const Tensor<N>> &tensor) {
    return getTensorDataPtr(*tensor);
  }

  template <int N>
  static auto getTensorDataPtr(Tensor<N> &tensor) {
    return tensor.data_;
  }

  template <int N>
  static auto getTensorDataPtr(const std::shared_ptr<Tensor<N>> &tensor) {
    return getTensorDataPtr(*tensor);
  }
};
}  // namespace dllm