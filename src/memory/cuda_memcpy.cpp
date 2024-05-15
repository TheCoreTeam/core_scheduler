#include "memory/cuda_memcpy.h"

#include "util.h"

namespace dllm::memory {
TaskCudart memcpyFromHost(const std::shared_ptr<Tensor1D>& dst,
                          const void* src) {
  auto task = TaskCudart{[dst = dst, src = src, future = *dst->future](
                             const ContextCudart* context) mutable {
    const auto size = cute::size(dst->layout);
    util::FutureGuard guard{future};
    CHECK_CUDART(cudaMemcpyAsync(dst->data(), src, toByte(dst->dtype) * size,
                                 cudaMemcpyHostToDevice, context->cudaStream));
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    dst.reset();
  }};
  *dst->future = task.get_future();
  return task;
}

TaskCudart memcpyToHost(void* dst, const std::shared_ptr<const Tensor1D>& src) {
  auto task = TaskCudart{[src = src, dst = dst, future = *src->future](
                             const ContextCudart* context) mutable {
    const auto size = cute::size(src->layout);
    util::FutureGuard guard{future};
    CHECK_CUDART(cudaMemcpyAsync(dst, src->data(), toByte(src->dtype) * size,
                                 cudaMemcpyDeviceToHost, context->cudaStream));
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    src.reset();
  }};
  *src->future = task.get_future();
  return task;
}
}  // namespace dllm::memory