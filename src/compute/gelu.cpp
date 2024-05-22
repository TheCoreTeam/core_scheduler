#include "compute/gelu.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute::GeLU {
void forwardKernel(cudaStream_t cudaStream, Tensor2D &output,
                   const Tensor2D &input);

void backwardKernel(cudaStream_t cudaStream, Tensor2D& dinput,
                    const Tensor2D& input, const Tensor2D& doutput);

TaskCompute forward(const std::shared_ptr<Tensor2D> &output,
                    const std::shared_ptr<const Tensor2D> &input) {
  if (output->layout.shape<0>() != input->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task = TaskCompute{
      [output=output, input=input,
     outputfuture = *output->future, inputFuture = input->future->wFuture](const ContextCompute *context) mutable{
    {
      util::FutureGuard outputrGuard{outputfuture.rFuture};
      util::FutureGuard outputwGuard{outputfuture.wFuture};
      util::FutureGuard inputGuard{inputFuture};
      forwardKernel(context->cudaStream, *output, *input);
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    output.reset();
    input.reset();
  }};
  const TaskFuture future = task.get_future();
  input->future->rFuture = future;
  output->future->wFuture = future;
  return task;
}

TaskCompute backward(const std::shared_ptr<Tensor2D> &dinput,
                    const std::shared_ptr<const Tensor2D> &input,
                    const std::shared_ptr<const Tensor2D> &doutput
                     ) {
  if (doutput->layout.shape<1>() != input->layout.shape<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task = TaskCompute{
      [doutput=doutput, input=input,dinput=dinput,
       dinputfuture = *dinput->future, doutputfuture = doutput->future->wFuture, inputFuture = input->future->wFuture](const ContextCompute *context) mutable{
        util::FutureGuard dinputRGuard{dinputfuture.rFuture};
        util::FutureGuard dinputWGuard{dinputfuture.wFuture};
        util::FutureGuard dyGuard{doutputfuture};
        util::FutureGuard wGuard{inputFuture};
        backwardKernel(context->cudaStream, *dinput, *input, *doutput);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        dinput.reset();
        input.reset();
        doutput.reset();
      }};
  const TaskFuture future = task.get_future();
  dinput->future->wFuture = future;
  doutput->future->rFuture = future;
  input->future->rFuture = future;
  return task;
}
}  // namespace dllm::compute::GeLU
