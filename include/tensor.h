#pragma once
#include <ATen/core/TensorBody.h>

#include <future>
#include <memory>
#include <type_traits>

#include "logger.h"
#include "threading/task_compute.h"

namespace dllm {
using at::IntArrayRef;

using at::TensorOptions;

enum TensorBackend { Undefined, Torch };

enum OpType { R = 1, W = R << 1, RW = R | W };

inline bool isR(const OpType op) { return op | R; }
inline bool isW(const OpType op) { return op | W; }
inline bool isRW(const OpType op) { return op & RW; }

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

  static bool valid() { return true; }
};

struct Tensor {
  Tensor() : future_{std::make_shared<TensorFuture>()} {}

  explicit Tensor(const at::Tensor &tensor)
      : tensor_{tensor}, future_{std::make_shared<TensorFuture>()} {}

  explicit Tensor(const at::Tensor &tensor,
                  const std::shared_ptr<TensorFuture> &future)
      : tensor_{tensor}, future_{future} {}

  [[nodiscard]] std::shared_ptr<Tensor> view(const IntArrayRef &size) const;

  struct TensorImpl;

  TensorFuture &future();

  const TensorFuture &future() const;

  void reset();

  void wait() const;

  operator at::Tensor() const;

  at::Tensor &tensor() { return tensor_; }

  const at::Tensor &tensor() const { return tensor_; }

  at::Tensor tensor_;

  std::shared_ptr<TensorFuture> future_;
};

template <typename Tensor>
constexpr TensorBackend declbackend() {
  using PlainTensor = std::decay_t<Tensor>;
  if constexpr (std::is_same_v<PlainTensor, at::Tensor>) {
    return Torch;
  } else {
    return Undefined;
  }
}

inline std::shared_ptr<Tensor> Tensor::view(const IntArrayRef &size) const {
  return std::make_shared<Tensor>(tensor_.view(size), future_);
}

inline TensorFuture &Tensor::future() { return *future_; }

inline const TensorFuture &Tensor::future() const { return *future_; }

inline void Tensor::wait() const { future().wait(); }

inline Tensor::operator at::Tensor() const { return tensor(); }
}  // namespace dllm
