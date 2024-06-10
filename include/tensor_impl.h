#pragma once
#ifdef DLLM_ENABLE_INTERNAL_BUILD
// This is an internal file, never use it unless you know what you are doing
#include <ATen/core/TensorBody.h>
#include <cuda_runtime.h>

#include <memory>

#include "tensor.h"
#include "threading/event.h"

namespace dllm {
struct Event;

struct ReadOnlyTensor::Impl {
  cudaStream_t stream_;
  auto &stream() { return stream_; }
  auto &stream() const { return stream_; }

  int8_t streamIdx_;
  auto &streamIdx() { return streamIdx_; }
  auto &streamIdx() const { return streamIdx_; }

  int8_t schedulerIdx_;
  auto &schedulerIdx() { return schedulerIdx_; }
  auto &schedulerIdx() const { return schedulerIdx_; }

  Event event_;
  auto &event() { return event_; }
  auto &event() const { return event_; }

  Impl() = default;

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

  at::Tensor tensor_{};

  IntArray sizes_{0};

  TensorOptions options_{};
};
}  // namespace dllm
#else
#error "You should not include this file in your program!"
#endif
