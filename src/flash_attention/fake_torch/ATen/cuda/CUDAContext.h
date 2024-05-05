#pragma once
#include <cuda_runtime.h>

#include <ATen/cuda/detail/PhiloxCudaStateRaw.cuh>
#include <ATen/cuda/detail/UnpackRaw.cuh>
#include <memory>

#include "logger.h"

#define C10_CUDA_CHECK(EXPR) CHECK_CUDART(EXPR)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

namespace at::cuda {
const cudaDeviceProp* getCurrentDeviceProperties();
}  // namespace at::cuda
