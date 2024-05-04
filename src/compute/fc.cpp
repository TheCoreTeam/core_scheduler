#include "compute/fc.h"

#include <spdlog/spdlog.h>

#include "compute/matmul.h"
#include "logger.h"
#include "util.h"

namespace dllm::compute::FcNoBias {
TaskCompute forward(const std::shared_ptr<Tensor2D> &y,
             const std::shared_ptr<const Tensor2D> &x,
             const std::shared_ptr<const Tensor2D> &w,
             const cublasComputeType_t computeType) {
  // y: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  return TaskCompute{
      [=, futureX = x->future, futureW = w->future](const ContextCompute *context) {
        util::waitFutureIfValid(futureX);
        util::waitFutureIfValid(futureW);
        RowMajorNTMatmulNoBias(context->cublasHandle, *x, *w, *y, computeType);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
}

TaskCompute backwardW(const std::shared_ptr<Tensor2D> &dw,
               const std::shared_ptr<const Tensor2D> &dy,
               const std::shared_ptr<const Tensor2D> &x,
               cublasComputeType_t computeType) {
  // dx, x: M * K
  // dy: M * N
  // dw = dy^T @ x
  return TaskCompute{[=, futureDy = dy->future,
               futureX = x->future](const ContextCompute *context) {
    util::waitFutureIfValid(futureDy);
    util::waitFutureIfValid(futureX);
    RowMajorTNMatmulNoBias(context->cublasHandle, *dy, *x, *dw, computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}

TaskCompute backwardX(const std::shared_ptr<Tensor2D> &dx,
               const std::shared_ptr<const Tensor2D> &dy,
               const std::shared_ptr<const Tensor2D> &w,
               cublasComputeType_t computeType) {
  // dw, w: N * K
  // dy: M * N
  // dx = dy @ w
  return TaskCompute{[=, futureDy = dy->future,
               futureW = w->future](const ContextCompute *context) {
    util::waitFutureIfValid(futureDy);
    util::waitFutureIfValid(futureW);
    RowMajorNNMatmulNoBias(context->cublasHandle, *dy, *w, *dx, computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}
}  // namespace dllm::compute::FcNoBias
