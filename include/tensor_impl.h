#pragma once
#ifdef DLLM_ENABLE_INTERNAL_BUILD
// This is an internal file, never use it unless you know what you are doing
#include <ATen/core/TensorBody.h>
#include <cuda_runtime.h>

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

  [[nodiscard]] auto options() const { return tensor_.options(); }

  [[nodiscard]] auto sizes() const { return tensor_.sizes(); }

  [[nodiscard]] auto size(const int64_t dim) const { return tensor_.size(dim); }

  [[nodiscard]] auto numel() const { return tensor_.numel(); }

  auto &tensor() { return tensor_; }

  at::Tensor tensor_{};
};
}  // namespace dllm
#else
#error "You should not include this file in your program!"
#endif
