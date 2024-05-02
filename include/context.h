#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace dllm {
struct Context {
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
};
}  // namespace dllm