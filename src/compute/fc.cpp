#include "compute/fc.h"

#include <spdlog/spdlog.h>

#include "compute/matmul.h"
#include "logger.h"

namespace dllm::compute::FcNoBias {
Task forward(const std::shared_ptr<Tensor2D> &y,
             const std::shared_ptr<const Tensor2D> &x,
             const std::shared_ptr<const Tensor2D> &w,
             const cublasComputeType_t computeType) {
  // y: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  return Task{[=](const Context *context) {
    x->waitFutureIfValid();
    w->waitFutureIfValid();
    RowMajorNTMatmulNoBias(context->cublasHandle, *x, *w, *y, computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}

Task backwardW(const std::shared_ptr<Tensor2D> &dw,
               const std::shared_ptr<const Tensor2D> &dy,
               const std::shared_ptr<const Tensor2D> &x,
               cublasComputeType_t computeType) {
  // dx, x: M * K
  // dy: M * N
  // dw = dy^T @ x
  return Task{[=](const Context *context) {
    dy->waitFutureIfValid();
    x->waitFutureIfValid();
    RowMajorTNMatmulNoBias(context->cublasHandle, *dy, *x, *dw, computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}

Task backwardX(const std::shared_ptr<Tensor2D> &dx,
               const std::shared_ptr<const Tensor2D> &dy,
               const std::shared_ptr<const Tensor2D> &w,
               cublasComputeType_t computeType) {
  // dw, w: N * K
  // dy: M * N
  // dx = dy @ w
  return Task{[=](const Context *context) {
    dy->waitFutureIfValid();
    w->waitFutureIfValid();
    RowMajorNNMatmulNoBias(context->cublasHandle, *dy, *w, *dx, computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}
}  // namespace dllm::compute::FcNoBias
