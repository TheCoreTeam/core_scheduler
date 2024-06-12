#pragma once
#include <ATen/cuda/CUDAEvent.h>

#include "threading/event.h"

namespace dllm {
struct Event::Impl {
  void block();

  void record();

  [[nodiscard]] bool query() const;

  void synchronize() const;

 private:
  at::cuda::CUDAEvent event;
};
}  // namespace dllm
