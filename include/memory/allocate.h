#pragma once
#include "tensor.h"
#include "threading/task_compute.h"
#include "threading/task_cudart.h"

namespace dllm::memory {
template <int N>
TaskCompute allocateRowMajor(
    std::shared_ptr<Tensor<N>> &p,
    const typename repeat_type<TensorIndexType, N, cute::Shape>::type &shape,
    const Dtype &dtype, const DeviceType &deviceType);
}  // namespace dllm::memory
