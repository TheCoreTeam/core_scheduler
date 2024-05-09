#include "compute/random.h"
#include "logger.h"

namespace dllm::compute::Random {
void kaimingNormKernel(const ContextCompute *context, Tensor2D &x,
                       double stddev);

TaskCompute kaimingNorm(const std::shared_ptr<Tensor2D> &x) {
  double stddev = std::sqrt(2.0f / cute::shape<1>(x->layout));
  return TaskCompute{[=](const ContextCompute *context) {
    kaimingNormKernel(context, *x, stddev);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}
}  // namespace dllm::compute::Random
