#include "memory/cuda_memcpy.h"

namespace dllm::memory {
TaskCudart memcpyFromHost(const std::shared_ptr<Tensor1D>& dst,
                          const void* src) {
  auto task = TaskCudart{[=,
                          future = *dst->future](const ContextCudart* context) {
    const auto size = cute::size(dst->layout);
    CHECK_CUDART(cudaMemcpyAsync(dst->data(), src, toByte(dst->dtype) * size,
                                 cudaMemcpyHostToDevice, context->cudaStream));
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  *dst->future = task.get_future();
  return task;
}

TaskCudart memcpyToHost(void* dst, const std::shared_ptr<const Tensor1D>& src) {
  auto task = TaskCudart{[=,
                          future = *src->future](const ContextCudart* context) {
    const auto size = cute::size(src->layout);
    CHECK_CUDART(cudaMemcpyAsync(dst, src->data(), toByte(src->dtype) * size,
                                 cudaMemcpyDeviceToHost, context->cudaStream));
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  *src->future = task.get_future();
  return task;
}
}  // namespace dllm::memory