#pragma once
#include <cublas_v2.h>

#include "tensor.h"

namespace dllm {
struct FcNoBias {
  static void forward(cublasHandle_t handle, const Tensor3D &y,
                      const Tensor3D &x, const Tensor2D &w,
                      cublasComputeType_t computeType);

  static void backwardW(cublasHandle_t handle, const Tensor2D &dw,
                        const Tensor3D &dy, const Tensor3D &x,
                        cublasComputeType_t computeType);

  static void backwardX(cublasHandle_t handle, const Tensor3D &dx,
                        const Tensor3D &dy, const Tensor2D &w,
                        cublasComputeType_t computeType);
};
}  // namespace dllm