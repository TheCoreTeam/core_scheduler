#pragma once
#include <ATen/core/TensorBody.h>
#include <torch/cuda.h>

#include <exception>
#include <future>
#include <memory>

#include "threading/task_future.h"

namespace dllm {
struct TensorFriend;

using IntArrayRef = at::IntArrayRef;

using IntArray = c10::SmallVector<IntArrayRef::value_type>;

using TensorOptions = at::TensorOptions;

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

struct ReadOnlyTensor {
  static auto create() {
    return std::shared_ptr<const ReadOnlyTensor>(nullptr);
  }

  [[nodiscard]] auto &future() { return future_->wFuture; }

  [[nodiscard]] const auto &future() const { return future_->wFuture; }

  void resetFuture(const TaskFuture &future) const {
    future_->rFuture = future;
  }

  [[nodiscard]] IntArray &sizes() { return sizes_; }

  [[nodiscard]] const IntArray &sizes() const { return sizes_; }

  [[nodiscard]] auto size(const int64_t dim) const {
    return dim >= 0 ? sizes()[dim] : sizes()[sizes().size() + dim];
  }

  void wait() const {
    if (future().valid()) {
      future().wait();
    }
  }

 protected:
  ReadOnlyTensor() : future_{std::make_shared<TensorFuture>()} {}

  explicit ReadOnlyTensor(const at::Tensor &tensor,
                          const std::shared_ptr<TensorFuture> &future)
      : tensor_{tensor}, future_{future} {}

  explicit ReadOnlyTensor(const std::shared_ptr<TensorFuture> &future)
      : future_{future} {}

  at::Tensor tensor_{};

  std::shared_ptr<TensorFuture> future_{};

  IntArray sizes_{0};

#ifdef DLLM_ENABLE_INTERNAL_BUILD
  friend TensorFriend;
#endif
};

struct Tensor : ReadOnlyTensor {
  static auto create() { return std::make_shared<Tensor>(Tensor{}); }

  [[nodiscard]] auto &future() { return *future_; }

  [[nodiscard]] const auto &future() const { return *future_; }

  void resetFuture(const TaskFuture &future) const {
    future_->wFuture = future;
  }

  void wait() const {
    if (future().valid()) {
      future().wait();
      try {
        future().get();
      } catch (const std::exception &) {
        std::rethrow_exception(std::current_exception());
      }
    }
  }

 private:
  Tensor() = default;

  explicit Tensor(const at::Tensor &tensor,
                  const std::shared_ptr<TensorFuture> &future)
      : ReadOnlyTensor{tensor, future} {}

  explicit Tensor(const std::shared_ptr<TensorFuture> &future)
      : ReadOnlyTensor{future} {}

#ifdef DLLM_ENABLE_INTERNAL_BUILD
  friend TensorFriend;
#endif
};
}  // namespace dllm

namespace at {
bool allclose(const std::shared_ptr<const ::dllm::ReadOnlyTensor> &t1,
              const at::Tensor &t2, double rtol = 1e-05, double atol = 1e-08,
              bool equal_nan = false);
bool allclose(const std::shared_ptr<const ::dllm::ReadOnlyTensor> &t1,
              const std::shared_ptr<const ::dllm::ReadOnlyTensor> &t2,
              double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
bool allclose(const at::Tensor &t1,
              const std::shared_ptr<const ::dllm::ReadOnlyTensor> &t2,
              double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
}  // namespace at
