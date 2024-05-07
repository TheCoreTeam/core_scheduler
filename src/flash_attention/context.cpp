#include <cuda_runtime.h>

#include "ATen/cuda/CUDAContext.h"
#include "logger.h"

namespace at::cuda {
namespace {
struct cudaDevicePropWarpper {
  std::unique_ptr<cudaDeviceProp[]> prop{[] {
    int deviceCount;
    CHECK_CUDART(cudaGetDeviceCount(&deviceCount));
    auto prop = new cudaDeviceProp[deviceCount];
    for (int i = 0; i < deviceCount; ++i) {
      CHECK_CUDART(cudaGetDeviceProperties_v2(&prop[i], i));
    }
    return prop;
  }()};
};
}  // namespace

cudaDeviceProp* getCurrentDeviceProperties() {
  static cudaDevicePropWarpper prop{};
  int device;
  CHECK_CUDART(cudaGetDevice(&device));
  return &prop.prop.get()[device];
}
}  // namespace at::cuda
