#include "allocate/malloc_from_mem_pool.h"

namespace dllm::alloc {
void mallocFromMemPool(std::shared_ptr<Tensor1D> &x, cudaStream_t stream) {
  if (x->data() != nullptr) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "The data must be nullptr");
  }
  x->resetData(std::shared_ptr<void>{
      [&] {
        void *ptr;
        CHECK_CUDART(cudaMallocAsync(&ptr, cute::size(x->layout), stream));
        return ptr;
      }(),
      [=](void *ptr) { CHECK_CUDART(cudaFreeAsync(ptr, stream)); }});
}
}  // namespace dllm::alloc
