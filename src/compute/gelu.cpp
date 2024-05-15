#include "compute/gelu.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute::GeLU {
void forwardKernel(cudaStream_t cudaStream, Tensor1D &output,
                   const Tensor1D &input);

TaskCompute forward(const std::shared_ptr<Tensor1D> &output,
                    const std::shared_ptr<const Tensor1D> &input) {
  if (output->layout.shape<0>() != input->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task = TaskCompute{
      [output = output, input = input, outputFuture = *output->future,
       inputFuture = *input->future](const ContextCompute *context) mutable {
        {
          util::FutureGuard outputGuard{outputFuture};
          util::FutureGuard inputGuard{inputFuture};
          forwardKernel(context->cudaStream, *output, *input);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        output.reset();
        input.reset();
      }};
  *input->future = task.get_future();
  return task;
}
}  // namespace dllm::compute::GeLU
