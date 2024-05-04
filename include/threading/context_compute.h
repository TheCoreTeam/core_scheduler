#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace dllm {
struct ContextCompute {
  cudaStream_t cudaStream{nullptr};
  cublasHandle_t cublasHandle{nullptr};
};
}  // namespace dllm