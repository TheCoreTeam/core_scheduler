#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <mutex>

namespace dllm {
struct ContextCompute {
  int deviceRank{0};
  // the random state will be changed all the time
  mutable curandState_t curandState{};
  mutable std::mutex curandStateMutex{};
  cudaStream_t cudaStream{nullptr};
  cudaMemPool_t memPool{nullptr};
  cublasHandle_t cublasHandle{nullptr};
};
}  // namespace dllm