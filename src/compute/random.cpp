#include "compute/random.h"

namespace dllm::compute::Random {
void gaussianKernel(const ContextCompute *context, Tensor1D &tensor);

void uniformKernel(const ContextCompute *context, Tensor1D &tensor);

TaskCompute gaussian(const std::shared_ptr<Tensor1D> &x) {
  return TaskCompute{[=](const ContextCompute *context) {
    gaussianKernel(context, *x);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}

TaskCompute uniform(const std::shared_ptr<Tensor1D> &x) {
  return TaskCompute{[=](const ContextCompute *context) {
    uniformKernel(context, *x);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}
}  // namespace dllm::compute::Random
