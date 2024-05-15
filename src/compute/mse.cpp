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
                   xFuture = *x->future, yFuture = *y->future](
                      const ContextCompute *context) mutable {
        {
          util::FutureGuard errorGuard{errorFuture};
          util::FutureGuard xGuard{xFuture};
          util::FutureGuard yGuard{yFuture};
          forwardKernel(context->cudaStream, *error, *x, *y);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        error.reset();
        x.reset();
        y.reset();
      }};
  const auto &future = *error->future = task.get_future();
  *x->future = future;
  *y->future = future;
  return task;
}

TaskCompute backward(const std::shared_ptr<Tensor1D> &dx,
                     const std::shared_ptr<const Tensor1D> &x,
                     const std::shared_ptr<const Tensor1D> &y) {
  if (x->layout.shape<0>() != y->layout.shape<0>() ||
      x->layout.shape<0>() != dx->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task = TaskCompute{
      [dx = dx, x = x, y = y, dxFuture = *dx->future, xFuture = *x->future,
       yFuture = *y->future](const ContextCompute *context) mutable {
        {
          util::FutureGuard dxGuard{dxFuture};
          util::FutureGuard xGuard{xFuture};
          util::FutureGuard yGuard{yFuture};
          backwardKernel(context->cudaStream, *dx, *x, *y);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        dx.reset();
        x.reset();
        y.reset();
      }};
  const auto &future = *dx->future = task.get_future();
  *x->future = future;
  *y->future = future;
  return task;
}
}  // namespace dllm::compute::Mse