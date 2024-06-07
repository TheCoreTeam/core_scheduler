#pragma once
#ifdef DLLM_ENABLE_INTERNAL_BUILD
// This is an internal file, never use it unless you know what you are doing
#include <ATen/core/TensorBody.h>

#include <memory>

#include "tensor.h"
#include "threading/task_future.h"

namespace dllm {
struct ReadOnlyTensor::Impl {
  struct TensorFuture {
    mutable TaskFuture rFuture{};
    mutable TaskFuture wFuture{};

    void wait() const {
      if (rFuture.valid()) {
        rFuture.wait();
      }
      if (wFuture.valid()) {
        wFuture.wait();
      }
    }

    void get() const {
      if (rFuture.valid()) {
        rFuture.get();
      }
      if (wFuture.valid()) {
        wFuture.get();
      }
    }

    static bool valid() { return true; }
  };

  Impl() = default;

  explicit Impl(const at::Tensor &tensor,
                const std::shared_ptr<TensorFuture> &future)
      : tensor_{tensor}, future_{future} {}

  explicit Impl(const std::shared_ptr<TensorFuture> &future)
      : future_{future} {}

  void resetFuture(const TaskFuture &future) const {
    future_->rFuture = future;
  }

  [[nodiscard]] const TensorOptions &options() const { return options_; }

  [[nodiscard]] TensorOptions &options() { return options_; }

  [[nodiscard]] IntArray &sizes() { return sizes_; }

  [[nodiscard]] const IntArray &sizes() const { return sizes_; }

  [[nodiscard]] auto size(const int64_t dim) const {
    return dim >= 0 ? sizes()[dim] : sizes()[sizes().size() + dim];
  }

  [[nodiscard]] auto numel() const {
    int64_t c = 1;
    for (const auto s : sizes()) {
      c *= s;
    }
    return c;
  }

  auto &tensor() { return tensor_; }

  auto &futurePtr() { return future_; }

  auto &tensor() const { return tensor_; }

  auto &futurePtr() const { return future_; }

  at::Tensor tensor_{};

  std::shared_ptr<TensorFuture> future_ = std::make_shared<TensorFuture>();

  IntArray sizes_{0};

  TensorOptions options_{};
};

namespace utils {
inline auto future(const ReadOnlyTensor &tensor) {
  return tensor.impl()->future_->wFuture;
}

inline auto future(const Tensor &tensor) { return *tensor.impl()->future_; }

inline void resetFuture(const ReadOnlyTensor &tensor,
                        const TaskFuture &future) {
  tensor.impl()->future_->rFuture = future;
}

inline void resetFuture(const Tensor &tensor, const TaskFuture &future) {
  tensor.impl()->future_->wFuture = future;
}
}  // namespace utils
}  // namespace dllm
#else
#error "You should not include this file in your program!"
#endif
