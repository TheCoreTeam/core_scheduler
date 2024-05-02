#pragma once
#include <cublas_v2.h>

#include "tensor.h"

namespace dllm::compute::FcNoBias {
static Task forward(const std::shared_ptr<Tensor3D> &y,
                    const std::shared_ptr<const Tensor3D> &x,
                    const std::shared_ptr<const Tensor2D> &w,
                    cublasComputeType_t computeType);

static Task backwardW(const std::shared_ptr<Tensor2D> &dw,
                      const std::shared_ptr<const Tensor3D> &dy,
                      const std::shared_ptr<const Tensor3D> &x,
                      cublasComputeType_t computeType);

static Task backwardX(const std::shared_ptr<Tensor3D> &dx,
                      const std::shared_ptr<const Tensor3D> &dy,
                      const std::shared_ptr<const Tensor2D> &w,
                      cublasComputeType_t computeType);
}  // namespace dllm::compute::FcNoBias
