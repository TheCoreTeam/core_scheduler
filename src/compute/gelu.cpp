#include "compute/gelu.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute {
void GeLUKernel(cudaStream_t cudaStream, Tensor1D &output,
                const Tensor1D &input);

TaskCompute GeLU(const std::shared_ptr<Tensor1D> &output,
                 const std::shared_ptr<const Tensor1D> &input) {
  if (output->layout.shape<0>() != input->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task = TaskCompute{
      [=, futureInput = *input->future](const ContextCompute *context) {
        util::waitFutureIfValid(futureInput);
        GeLUKernel(context->cudaStream, *output, *input);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  *input->future = task.get_future();
  return task;
}
}  // namespace dllm::compute
