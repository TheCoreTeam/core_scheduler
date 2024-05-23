#include "compute/add.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute::Add {

void forwardKernel(cudaStream_t stream, Tensor3D& output, const Tensor3D& A,
                   const Tensor3D& B);

void backwardKernel(cudaStream_t stream, Tensor3D& grad_A, Tensor3D& grad_B,
                    const Tensor3D& grad_output);

TaskCompute forward(const std::shared_ptr<Tensor3D>& output,
                    const std::shared_ptr<const Tensor3D>& A,
                    const std::shared_ptr<const Tensor3D>& B) {
  if (!(output->layout == A->layout && output->layout == B->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  if (output->dtype != A->dtype && output->dtype != B->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data type not same");
  }
  if (output->deviceType != dllm::CUDA && A->deviceType != dllm::CUDA &&
      B->deviceType != dllm::CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input are not on CUDA device");
  }
  auto task = TaskCompute{
      [output = output, A = A, B = B, AFuture = A->future->wFuture,
       BFuture = B->future->wFuture,
       outputFuture = *output->future](const ContextCompute* context) mutable {
        {
          util::FutureGuard AGuard{AFuture};
          util::FutureGuard BGuard{BFuture};
          util::FutureGuard outputRGuard{outputFuture.rFuture};
          util::FutureGuard outputWGuard{outputFuture.wFuture};
          forwardKernel(context->cudaStream, *output, *A, *B);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        A.reset();
        B.reset();
        output.reset();
      }};
  const TaskFuture future = task.get_future();
  A->future->rFuture = future;
  B->future->rFuture = future;
  output->future->wFuture = future;
  return task;
}

TaskCompute backward(const std::shared_ptr<Tensor3D>& grad_A,
                     const std::shared_ptr<Tensor3D>& grad_B,
                     const std::shared_ptr<const Tensor3D>& grad_output) {
  if (!(grad_output->layout == grad_A->layout &&
        grad_output->layout == grad_B->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  if (grad_output->dtype != grad_A->dtype &&
      grad_output->dtype != grad_B->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data type not same");
  }
  if (grad_output->deviceType != dllm::CUDA &&
      grad_A->deviceType != dllm::CUDA && grad_B->deviceType != dllm::CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input are not on CUDA device");
  }
  auto task =
      TaskCompute{[grad_A = grad_A, grad_B = grad_B, grad_output = grad_output,
                   grad_outputFuture = grad_output->future->rFuture,
                   grad_inputFuture = *grad_A->future,
                   grad_residualFuture =
                       *grad_B->future](const ContextCompute* context) mutable {
        {
      util::FutureGuard grad_outputGuard{grad_outputFuture};
      util::FutureGuard grad_inputRGuard{grad_inputFuture.rFuture};
      util::FutureGuard grad_inputWGuard{grad_inputFuture.wFuture};
      util::FutureGuard grad_residualRGuard{grad_residualFuture.rFuture};
      util::FutureGuard grad_residualWGuard{grad_residualFuture.wFuture};
      backwardKernel(context->cudaStream, *grad_A, *grad_B, *grad_output);
        }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    grad_output.reset();
        grad_A.reset();
        grad_B.reset();
      }};
  const TaskFuture future = task.get_future();
  grad_output->future->rFuture = future;
  grad_A->future->wFuture = future;
  grad_B->future->wFuture = future;
  return task;
}
}  // namespace dllm::compute::Residual
