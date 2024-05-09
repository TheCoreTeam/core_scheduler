#include "allocate/malloc_from_mem_pool.h"

#include "allocate/malloc_from_mem_internal.h"

namespace dllm::alloc {
void mallocFromMemPool(std::shared_ptr<Tensor1D> &x, cudaStream_t stream) {
  if (x->data() != nullptr) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "The data must be nullptr");
  }
  if (x->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(
        &logger(),
        "This is a CUDA function but the tensor is not a CUDA tensor");
  }
  x->resetData(std::shared_ptr<void>{
      [&] {
        void *ptr;
        CHECK_CUDART(cudaMallocAsync(
            &ptr, toByte(x->dtype) * cute::size(x->layout), stream));
        CHECK_CUDART(cudaStreamSynchronize(stream));
        return ptr;
      }(),
      [=](void *ptr) { CHECK_CUDART(cudaFreeAsync(ptr, stream)); }});
}

std::shared_ptr<void> mallocFromMemPool(size_t size, Dtype dtype,
                                        const ContextCompute *context) {
  return std::shared_ptr<void>{
      [&] {
        void *ptr;
        CHECK_CUDART(cudaMallocFromPoolAsync(
            &ptr, toByte(dtype) * size, context->memPool, context->cudaStream));
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        return ptr;
      }(),
      [stream = context->cudaStream](void *ptr) {
        CHECK_CUDART(cudaFreeAsync(ptr, stream));
      }};
}
}  // namespace dllm::alloc
