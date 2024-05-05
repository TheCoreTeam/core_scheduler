#pragma once
#include <cuda_runtime.h>

#include <ATen/cuda/detail/PhiloxCudaStateRaw.cuh>
#include <ATen/cuda/detail/UnpackRaw.cuh>
#include <memory>

#include "logger.h"

#define C10_CUDA_CHECK(EXPR) CHECK_CUDART(EXPR)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

namespace at::cuda {
std::unique_ptr<cudaDeviceProp> getCurrentDeviceProperties() {
  int device;
  CHECK_CUDART(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CHECK_CUDART(cudaGetDeviceProperties_v2(&prop, device));
  return std::make_unique<cudaDeviceProp>(prop);
}
}  // namespace at::cuda
