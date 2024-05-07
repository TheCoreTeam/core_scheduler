#pragma once
#include <cuda_runtime.h>

namespace dllm {
enum Dtype { R_64F, R_32F, R_16F, R_16BF, R_8U, R_32I };

__inline__ __attribute__((always_inline)) std::size_t toByte(Dtype dtype) {
  switch (dtype) {
    case R_64F:
      return 8;
    case R_32I:
    case R_32F:
      return 4;
    case R_16F:
    case R_16BF:
      return 2;
    case R_8U:
      return 1;
    default:
      return 0;
  }
}

template <typename T>
constexpr Dtype toDtype() {
  if constexpr (std::is_same_v<T, double>) {
    return R_64F;
  } else if constexpr (std::is_same_v<T, float>) {
    return R_32F;
  } else if constexpr (std::is_same_v<T, nv_half>) {
    return R_16F;
  } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
    return R_16BF;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return R_32I;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return R_8U;
  }
}

__inline__ __attribute__((always_inline)) cudaDataType
toCudaDataType(Dtype dtype) {
  switch (dtype) {
    case R_64F:
      return CUDA_R_64F;
    case R_32F:
      return CUDA_R_32F;
    case R_16F:
      return CUDA_R_16F;
    case R_16BF:
      return CUDA_R_16BF;
    case R_32I:
      return CUDA_R_32I;
    case R_8U:
      return CUDA_R_8U;
    default:
      return static_cast<cudaDataType>(-1);
  }
}
}  // namespace dllm