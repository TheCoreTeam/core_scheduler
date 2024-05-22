#include "memory/malloc_from_mem_internal.h"
#include "tensor_friend.h"

namespace dllm::memory {
void mallocFromMemPool(const std::shared_ptr<Tensor1D> &x,
                       const cudaStream_t stream) {
  if (x->data() != nullptr) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "The data must be nullptr");
  }
  if (x->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(
        &logger(),
        "This is a CUDA function but the tensor is not a CUDA tensor");
  }
  auto p = Tensor1D::DataPtr{
      [&] {
        void *ptr;
        CHECK_CUDART(cudaMallocAsync(
            &ptr, toByte(x->dtype) * cute::size(x->layout), stream));
        CHECK_CUDART(cudaStreamSynchronize(stream));
        return ptr;
      }(),
      [=](const void *ptr) {
        CHECK_CUDART(cudaFreeAsync(const_cast<void *>(ptr), stream));
      }};
  TensorFriend::resetTensorData(x, std::move(p));
}

Tensor1D::DataPtr mallocFromMemPool(const size_t size, const Dtype dtype,
                                    const ContextCompute *context) {
  return {
      [&] {
        void *ptr;
        CHECK_CUDART(cudaMallocFromPoolAsync(
            &ptr, toByte(dtype) * size, context->memPool, context->cudaStream));
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        return ptr;
      }(),
      [stream = context->cudaStream](const void *ptr) {
        CHECK_CUDART(cudaFreeAsync(const_cast<void *>(ptr), stream));
      }};
}
}  // namespace dllm::memory
