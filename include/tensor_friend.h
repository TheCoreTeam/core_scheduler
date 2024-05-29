#pragma once
#ifdef DLLM_ENABLE_INTERNAL_BUILD
// This is an internal file, never use it unless you know what you are doing
#include <memory>

#include "tensor.h"

namespace dllm {
struct TensorFriend {
  template <typename... Args>
  static std::shared_ptr<Tensor> create(Args &&...args) {
    return std::make_shared<Tensor>(Tensor{std::forward<Args>(args)...});
  }

  template <typename... Args>
  static std::shared_ptr<const ReadOnlyTensor> create_read_only(
      Args &&...args) {
    return std::make_shared<const ReadOnlyTensor>(
        ReadOnlyTensor{std::forward<Args>(args)...});
  }

  static auto &extract_future_ptr(
      const std::shared_ptr<const ReadOnlyTensor> &tensor) {
    return tensor->future_;
  }

  static auto &extract_future_ptr(const std::shared_ptr<Tensor> &tensor) {
    return tensor->future_;
  }

  static auto &extract_tensor(const std::shared_ptr<Tensor> &tensor) {
    return tensor->tensor_;
  }

  static auto &extract_tensor(
      const std::shared_ptr<const ReadOnlyTensor> &tensor) {
    return tensor->tensor_;
  }
};

#define DLLM_EXTRACT_TENSOR(tensor) \
  ::dllm::TensorFriend::extract_tensor((tensor))
}  // namespace dllm
#else
#error "You should not include this file in your program!"
#endif
