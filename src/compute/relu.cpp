#include "compute/relu.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute::ReLU {
void forwardKernel(cudaStream_t stream, const Tensor1D& input,
                   Tensor1D& output);

TaskCompute forward(const std::shared_ptr<const Tensor1D>& input,
                    const std::shared_ptr<Tensor1D>& output) {
  if (output->layout.shape<0>() != input->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task = TaskCompute{
      [=, futureInput = *input->future](const ContextCompute* context) {
        util::waitFutureIfValid(futureInput);
        forwardKernel(context->cudaStream, *input, *output);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  *input->future = task.get_future();
  return task;
}
}  // namespace dllm::compute::ReLU
