#pragma once
#include <cuda_runtime.h>

namespace dllm {
struct ContextCudart {
  int deviceRank{0};
  cudaMemPool_t memPool{nullptr};
  cudaStream_t cudaStream{nullptr};
};
}  // namespace dllm