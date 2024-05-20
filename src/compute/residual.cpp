#include "compute/residual.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute::Residual {

void forwardKernel(cudaStream_t stream, const Tensor3D& A, const Tensor3D& B,
                   Tensor3D& C);
void backwardKernel(cudaStream_t stream, const Tensor3D& grad_output,
                    Tensor3D& grad_A, Tensor3D& grad_B);

TaskCompute forward(const std::shared_ptr<const Tensor3D>& input,
                    const std::shared_ptr<const Tensor3D>& residual,
                    const std::shared_ptr<Tensor3D>& output) {
  if (!(output->layout == input->layout &&
        output->layout == residual->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  if (output->dtype != input->dtype && output->dtype != residual->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data type not same");
  }
  if (output->deviceType != dllm::CUDA && input->deviceType != dllm::CUDA &&
      residual->deviceType != dllm::CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input are not on CUDA device");
  }
  auto task = TaskCompute{
      [input = input, residual = residual, output = output,
       inputFuture = input->future->wFuture,
       residualFuture = residual->future->wFuture,
       outputFuture = *output->future](const ContextCompute* context) mutable {
        {
          util::FutureGuard inputGuard{inputFuture};
          util::FutureGuard residualGuard{residualFuture};
          util::FutureGuard outputRGuard{outputFuture.rFuture};
          util::FutureGuard outputWGuard{outputFuture.wFuture};
          forwardKernel(context->cudaStream, *input, *residual, *output);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        input.reset();
        residual.reset();
        output.reset();
      }};
  const TaskFuture future = task.get_future();
  input->future->rFuture = future;
  residual->future->rFuture = future;
  output->future->wFuture = future;
  return task;
}

TaskCompute backward(const std::shared_ptr<const Tensor3D>& grad_output,
                     const std::shared_ptr<Tensor3D>& grad_input,
                     const std::shared_ptr<Tensor3D>& grad_residual) {
  if (!(grad_output->layout == grad_input->layout &&
        grad_output->layout == grad_residual->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  if (grad_output->dtype != grad_input->dtype &&
      grad_output->dtype != grad_residual->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data type not same");
  }
  if (grad_output->deviceType != dllm::CUDA &&
      grad_input->deviceType != dllm::CUDA &&
      grad_residual->deviceType != dllm::CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input are not on CUDA device");
  }
  auto task = TaskCompute{[grad_output = grad_output, grad_input = grad_input,
                           grad_residual = grad_residual,
                           grad_outputFuture = grad_output->future->rFuture,
                           grad_inputFuture = *grad_input->future,
                           grad_residualFuture = *grad_residual->future](
                              const ContextCompute* context) mutable {
    {
      util::FutureGuard grad_outputGuard{grad_outputFuture};
      util::FutureGuard grad_inputRGuard{grad_inputFuture.rFuture};
      util::FutureGuard grad_inputWGuard{grad_inputFuture.wFuture};
      util::FutureGuard grad_residualRGuard{grad_residualFuture.rFuture};
      util::FutureGuard grad_residualWGuard{grad_residualFuture.wFuture};
      backwardKernel(context->cudaStream, *grad_output, *grad_input,
                     *grad_residual);
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    grad_output.reset();
    grad_input.reset();
    grad_residual.reset();
  }};
  const TaskFuture future = task.get_future();
  grad_output->future->rFuture = future;
  grad_input->future->wFuture = future;
  grad_residual->future->wFuture = future;
  return task;
}
}  // namespace dllm::compute::Residual
