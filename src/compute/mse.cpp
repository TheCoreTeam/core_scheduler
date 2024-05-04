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
  return TaskCompute{
      [=, futureX = x->future, futureY = y->future](const ContextCompute *context) {
        util::waitFutureIfValid(futureX);
        util::waitFutureIfValid(futureY);
        forwardKernel(context->cudaStream, *error, *x, *y);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
}

TaskCompute backward(const std::shared_ptr<Tensor1D> &dx,
              const std::shared_ptr<const Tensor1D> &x,
              const std::shared_ptr<const Tensor1D> &y) {
  if (x->layout.shape<0>() != y->layout.shape<0>() ||
      x->layout.shape<0>() != dx->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  return TaskCompute{
      [=, futureX = x->future, futureY = y->future](const ContextCompute *context) {
        util::waitFutureIfValid(futureX);
        util::waitFutureIfValid(futureY);
        backwardKernel(context->cudaStream, *dx, *x, *y);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
}
}  // namespace dllm::compute::Mse