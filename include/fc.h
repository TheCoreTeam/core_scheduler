#pragma once
#include <cublas_v2.h>

#include "tensor.h"

namespace dllm {
struct FcNoBias {
  static Task forward(const Tensor3D &y, const Tensor3D &x, const Tensor2D &w,
                      cublasComputeType_t computeType);

  static Task backwardW(const Tensor2D &dw, const Tensor3D &dy,
                        const Tensor3D &x, cublasComputeType_t computeType);

  static Task backwardX(const Tensor3D &dx, const Tensor3D &dy,
                        const Tensor2D &w, cublasComputeType_t computeType);
};
}  // namespace dllm