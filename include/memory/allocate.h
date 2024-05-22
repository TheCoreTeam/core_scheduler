#pragma once
#include "tensor.h"
#include "threading/task_compute.h"
#include "threading/task_cudart.h"

namespace dllm::memory {
template <int N>
  requires(N >= 1 && N <= 4)
TaskCompute allocate(std::shared_ptr<Tensor<N>> &p,
                     const typename Tensor<N>::Layout &layout,
                     const Dtype &dtype, const DeviceType &deviceType);

template <int N>
  requires(N >= 1 && N <= 4)
TaskCompute allocateRowMajor(std::shared_ptr<Tensor<N>> &p,
                             const typename Tensor<N>::Shape &shape,
                             const Dtype &dtype, const DeviceType &deviceType);
}  // namespace dllm::memory
