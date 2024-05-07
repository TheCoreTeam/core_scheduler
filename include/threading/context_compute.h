#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <mutex>

namespace dllm {
struct ContextCompute {
  int deviceRank{0};
  // the random state will be changed all the time
  unsigned long curandSeed{0};
  mutable std::atomic<unsigned long> curandOffset{0};
  cudaStream_t cudaStream{nullptr};
  cudaMemPool_t memPool{nullptr};
  cublasHandle_t cublasHandle{nullptr};
};
}  // namespace dllm