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
  auto task = TaskCompute{[computeType = computeType, x = x, w = w, y = y,
                           yFuture = *y->future, xFuture = x->future->wFuture,
                           wFuture = w->future->wFuture](
                              const ContextCompute *context) mutable {
    {
      util::FutureGuard yrGuard{yFuture.rFuture};
      util::FutureGuard ywGuard{yFuture.wFuture};
      util::FutureGuard xGuard{xFuture};
      util::FutureGuard wGuard{wFuture};
      RowMajorNTMatmulNoBias(context->cublasHandle, *x, *w, *y, computeType);
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    x.reset();
    w.reset();
    y.reset();
  }};
  const TaskFuture future = task.get_future();
  x->future->rFuture = future;
  w->future->rFuture = future;
  y->future->wFuture = future;
  return task;
}

TaskCompute backwardW(const std::shared_ptr<Tensor2D> &dw,
                      const std::shared_ptr<const Tensor2D> &dy,
                      const std::shared_ptr<const Tensor2D> &x,
                      cublasComputeType_t computeType) {
  // dx, x: M * K
  // dy: M * N
  // dw = dy^T @ x
  auto task = TaskCompute{
      [computeType = computeType, dy = dy, x = x, dw = dw,
       dwFuture = *dw->future, dyFuture = dy->future->wFuture,
       xFuture = x->future->wFuture](const ContextCompute *context) mutable {
        {
          util::FutureGuard dwRGuard{dwFuture.rFuture};
          util::FutureGuard dwWGuard{dwFuture.wFuture};
          util::FutureGuard dyGuard{dyFuture};
          util::FutureGuard xGuard{xFuture};
          RowMajorTNMatmulNoBias(context->cublasHandle, *dy, *x, *dw,
                                 computeType);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        dy.reset();
        x.reset();
        dw.reset();
      }};
  const TaskFuture future = task.get_future();
  dw->future->wFuture = future;
  dy->future->rFuture = future;
  x->future->rFuture = future;
  return task;
}

TaskCompute backwardX(const std::shared_ptr<Tensor2D> &dx,
                      const std::shared_ptr<const Tensor2D> &dy,
                      const std::shared_ptr<const Tensor2D> &w,
                      cublasComputeType_t computeType) {
  // dw, w: N * K
  // dy: M * N
  // dx = dy @ w
  auto task = TaskCompute{
      [computeType = computeType, dy = dy, w = w, dx = dx,
       dxFuture = *dx->future, dyFuture = dy->future->wFuture,
       wFuture = w->future->wFuture](const ContextCompute *context) mutable {
        {
          util::FutureGuard dxRGuard{dxFuture.rFuture};
          util::FutureGuard dxWGuard{dxFuture.wFuture};
          util::FutureGuard dyGuard{dyFuture};
          util::FutureGuard wGuard{wFuture};
          RowMajorNNMatmulNoBias(context->cublasHandle, *dy, *w, *dx,
                                 computeType);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        dy.reset();
        w.reset();
        dx.reset();
      }};
  const TaskFuture future = task.get_future();
  dx->future->wFuture = future;
  dy->future->rFuture = future;
  w->future->rFuture = future;
  return task;
}
}  // namespace dllm::compute::FcNoBias
