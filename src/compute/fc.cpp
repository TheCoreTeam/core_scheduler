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
  auto task =
      TaskCompute{[=, futureY = *y->future, futureX = *x->future,
                   futureW = *w->future](const ContextCompute *context) {
        util::waitFutureIfValid(futureY);
        util::waitFutureIfValid(futureX);
        util::waitFutureIfValid(futureW);
        RowMajorNTMatmulNoBias(context->cublasHandle, *x, *w, *y, computeType);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const auto &future = *y->future = task.get_future();
  *x->future = future;
  *w->future = future;
  return task;
}

TaskCompute backwardW(const std::shared_ptr<Tensor2D> &dw,
                      const std::shared_ptr<const Tensor2D> &dy,
                      const std::shared_ptr<const Tensor2D> &x,
                      cublasComputeType_t computeType) {
  // dx, x: M * K
  // dy: M * N
  // dw = dy^T @ x
  auto task = TaskCompute{[=, futureDw = *dw->future, futureDy = *dy->future,
                           futureX =
                               *x->future](const ContextCompute *context) {
    util::waitFutureIfValid(futureDw);
    util::waitFutureIfValid(futureDy);
    util::waitFutureIfValid(futureX);
    RowMajorTNMatmulNoBias(context->cublasHandle, *dy, *x, *dw, computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const auto &future = *dw->future = task.get_future();
  *dy->future = future;
  *x->future = future;
  return task;
}

TaskCompute backwardX(const std::shared_ptr<Tensor2D> &dx,
                      const std::shared_ptr<const Tensor2D> &dy,
                      const std::shared_ptr<const Tensor2D> &w,
                      cublasComputeType_t computeType) {
  // dw, w: N * K
  // dy: M * N
  // dx = dy @ w
  auto task = TaskCompute{[=, futureDx = *dx->future, futureDy = *dy->future,
                           futureW =
                               *w->future](const ContextCompute *context) {
    util::waitFutureIfValid(futureDx);
    util::waitFutureIfValid(futureDy);
    util::waitFutureIfValid(futureW);
    RowMajorNNMatmulNoBias(context->cublasHandle, *dy, *w, *dx, computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const auto &future = *dx->future = task.get_future();
  *dy->future = future;
  *w->future = future;
  return task;
}
}  // namespace dllm::compute::FcNoBias
