#include "compute/mse.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute::Mse {
void forwardKernel(cudaStream_t stream, Tensor1D &error, const Tensor1D &x,
                   const Tensor1D &y);

void backwardKernel(cudaStream_t stream, Tensor1D &dx, const Tensor1D &x,
                    const Tensor1D &y);

TaskCompute forward(const std::shared_ptr<Tensor1D> &error,
                    const std::shared_ptr<const Tensor1D> &x,
                    const std::shared_ptr<const Tensor1D> &y) {
  if (error->layout.shape<0>() != y->layout.shape<0>() ||
      x->layout.shape<0>() != y->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task =
      TaskCompute{[error = error, x = x, y = y, errorFuture = *error->future,
                   xFuture = x->future->wFuture, yFuture = y->future->rFuture](
                      const ContextCompute *context) mutable {
        {
          util::FutureGuard errorRGuard{errorFuture.rFuture};
          util::FutureGuard errorWGuard{errorFuture.wFuture};
          util::FutureGuard xGuard{xFuture};
          util::FutureGuard yGuard{yFuture};
          forwardKernel(context->cudaStream, *error, *x, *y);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        error.reset();
        x.reset();
        y.reset();
      }};
  const TaskFuture future = task.get_future();
  error->future->wFuture = future;
  x->future->rFuture = future;
  y->future->rFuture = future;
  return task;
}

TaskCompute backward(const std::shared_ptr<Tensor1D> &dx,
                     const std::shared_ptr<const Tensor1D> &x,
                     const std::shared_ptr<const Tensor1D> &y) {
  if (x->layout.shape<0>() != y->layout.shape<0>() ||
      x->layout.shape<0>() != dx->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task =
      TaskCompute{[dx = dx, x = x, y = y, dxFuture = *dx->future,
                   xFuture = x->future->wFuture, yFuture = y->future->wFuture](
                      const ContextCompute *context) mutable {
        {
          util::FutureGuard dxRGuard{dxFuture.rFuture};
          util::FutureGuard dxWGuard{dxFuture.wFuture};
          util::FutureGuard xGuard{xFuture};
          util::FutureGuard yGuard{yFuture};
          backwardKernel(context->cudaStream, *dx, *x, *y);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        dx.reset();
        x.reset();
        y.reset();
      }};
  const TaskFuture future = task.get_future();
  dx->future->wFuture = future;
  x->future->rFuture = future;
  y->future->rFuture = future;
  return task;
}
}  // namespace dllm::compute::Mse