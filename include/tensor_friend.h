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
};
}  // namespace dllm