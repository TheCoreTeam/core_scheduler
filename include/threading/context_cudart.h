#pragma once
#include <cuda_runtime.h>

namespace dllm {
struct ContextCudart {
  int deviceRank{0};
  cudaStream_t cudaStream{nullptr};
};
}  // namespace dllm