#include "compute/softmax.h"

#include "util.h"

namespace dllm::compute::SoftMax {
void forwardKernel(cudaStream_t stream, Tensor2D &output, const Tensor2D &input,
                   double scale);

TaskCompute forward(const std::shared_ptr<Tensor2D> &output,
                    const std::shared_ptr<const Tensor2D> &input,
                    const double scale) {
  if (output->layout != input->layout) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  if (output->dtype != input->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Dtype not same");
  }
  auto task = TaskCompute{
      [output = output, input = input, scale = scale,
       outputFuture = *output->future,
       inputFuture = *input->future](const ContextCompute *context) mutable {
        util::FutureGuard outputGuard{outputFuture};
        util::FutureGuard inputGuard{inputFuture};
        forwardKernel(context->cudaStream, *output, *input, scale);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        output.reset();
        input.reset();
      }};
  const auto &future = *output->future = task.get_future();
  *input->future = future;
  return task;
}
}  // namespace dllm::compute::SoftMax
