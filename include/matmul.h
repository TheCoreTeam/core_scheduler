#pragma once
#include "tensor.h"
#include <cublas_v2.h>

namespace dllm {
// C = A @ B
void RowMajorNNMatmulNoBias(cublasHandle_t handle, const Tensor2D &A,
                            const Tensor2D &B, const Tensor2D &C,
                            cublasComputeType_t computeType);

// C = A @ B^T
void RowMajorNTMatmulNoBias(cublasHandle_t handle, const Tensor2D &A,
                            const Tensor2D &B, const Tensor2D &C,
                            cublasComputeType_t computeType);

// C = A^T @ B
void RowMajorTNMatmulNoBias(cublasHandle_t handle, const Tensor2D &A,
                            const Tensor2D &B, const Tensor2D &C,
                            cublasComputeType_t computeType);
} // namespace dllm
