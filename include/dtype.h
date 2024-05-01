#pragma once
#include <cuda_runtime.h>

namespace dllm {
enum Dtype { R_64F, R_32F, R_16F, R_16BF };

template <typename T> constexpr Dtype toDtype() {
  if constexpr (std::is_same_v<T, double>) {
    return R_64F;
  } else if constexpr (std::is_same_v<T, float>) {
    return R_32F;
  } else if constexpr (std::is_same_v<T, nv_half>) {
    return R_16F;
  } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
    return R_16BF;
  }
}

inline cudaDataType toCudaDataType(Dtype dtype) {
  switch (dtype) {
  case R_64F:
    return CUDA_R_64F;
  case R_32F:
    return CUDA_R_32F;
  case R_16F:
    return CUDA_R_16F;
  case R_16BF:
    return CUDA_C_16BF;
  default:
    return static_cast<cudaDataType>(-1);
  }
}
} // namespace dllm