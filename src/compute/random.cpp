#include "compute/random.h"

#include "util.h"

namespace dllm::compute::Random {
void gaussianKernel(const ContextCompute *context, Tensor1D &tensor);

void uniformKernel(const ContextCompute *context, Tensor1D &tensor);

TaskCompute gaussian(const std::shared_ptr<Tensor1D> &x) {
  auto task = TaskCompute{
      [x = x, future = *x->future](const ContextCompute *context) mutable {
        util::FutureGuard rGuard{future.rFuture};
        util::FutureGuard wGuard{future.wFuture};
        gaussianKernel(context, *x);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        x.reset();
      }};
  const TaskFuture future = task.get_future();
  x->future->rFuture = future;
  x->future->wFuture = future;
  return task;
}

TaskCompute uniform(const std::shared_ptr<Tensor1D> &x) {
  auto task = TaskCompute{
      [x = x, future = *x->future](const ContextCompute *context) mutable {
        {
          util::FutureGuard rGuard{future.rFuture};
          util::FutureGuard wGuard{future.wFuture};
          uniformKernel(context, *x);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        x.reset();
      }};
  const TaskFuture future = task.get_future();
  x->future->rFuture = future;
  x->future->wFuture = future;
  return task;
}
}  // namespace dllm::compute::Random
