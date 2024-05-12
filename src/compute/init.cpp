#include "compute/random.h"
#include "logger.h"
#include "util.h"

namespace dllm::compute::Random {
void kaimingNormKernel(const ContextCompute *context, Tensor2D &x,
                       double stddev);

TaskCompute kaimingNorm(const std::shared_ptr<Tensor2D> &x) {
  double stddev = std::sqrt(2.0f / cute::shape<1>(x->layout));
  auto task =
      TaskCompute{[=, future = *x->future](const ContextCompute *context) {
        util::waitFutureIfValid(future);
        kaimingNormKernel(context, *x, stddev);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  *x->future = task.get_future();
  return task;
}
}  // namespace dllm::compute::Random
