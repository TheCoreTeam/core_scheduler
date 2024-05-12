#include "compute/random.h"

#include "util.h"

namespace dllm::compute::Random {
void gaussianKernel(const ContextCompute *context, Tensor1D &tensor);

void uniformKernel(const ContextCompute *context, Tensor1D &tensor);

TaskCompute gaussian(const std::shared_ptr<Tensor1D> &x) {
  auto task =
      TaskCompute{[=, future = *x->future](const ContextCompute *context) {
        util::waitFutureIfValid(future);
        gaussianKernel(context, *x);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  *x->future = task.get_future();
  return task;
}

TaskCompute uniform(const std::shared_ptr<Tensor1D> &x) {
  auto task =
      TaskCompute{[=, future = *x->future](const ContextCompute *context) {
        util::waitFutureIfValid(future);
        uniformKernel(context, *x);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  *x->future = task.get_future();
  return task;
}
}  // namespace dllm::compute::Random
