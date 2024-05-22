//
// Created by tingxuan on 2024/5/16.
//
#include "compute/embedding.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute::embedding {
void forwardKernel(cudaStream_t cudaStream, Tensor3D& output,
                   const Tensor2D& input, const Tensor2D& wte, const Tensor2D& wpe);
void backwardKernel(cudaStream_t cudaStream, const Tensor3D& grad_output,
                    Tensor2D& grad_input, Tensor2D& grad_wte, Tensor2D& grad_wpe);

TaskCompute forward(const std::shared_ptr<Tensor3D> &output,
                    const std::shared_ptr<const Tensor2D> &input,
                    const std::shared_ptr<const Tensor2D> &wte,
                    const std::shared_ptr<const Tensor2D> &wpe) {
  if (output->layout.shape<0>() != input->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task = TaskCompute{
      [input=input, output=output, wte= wte, wpe=wpe,futureOutput = *output->future,futureInput = input->future->wFuture,
       futureWte = wte->future->wFuture,futureWpe= wpe->future->wFuture](const ContextCompute *context) mutable{
        util::FutureGuard OutputrGuard{futureOutput.rFuture};
        util::FutureGuard OutputwGuard{futureOutput.wFuture};
        util::FutureGuard InputGuard{futureInput};
        util::FutureGuard wteGuard{futureWte};
        util::FutureGuard wpeGuard{futureWpe};

        forwardKernel(context->cudaStream, *output, *input, *wte, *wpe);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        input.reset();
        output.reset();
        wte.reset();
        wpe.reset();
      }};
  const TaskFuture future = task.get_future();
  input->future->rFuture = future;
  wte->future->rFuture = future;
  wpe->future->rFuture = future;
  output->future->wFuture = future;
  return task;
}

TaskCompute backward(const std::shared_ptr<const Tensor3D> &grad_output,
                    const std::shared_ptr<Tensor2D> &grad_input,
                    const std::shared_ptr<Tensor2D> &grad_wte,
                    const std::shared_ptr<Tensor2D> &grad_wpe) {
  if (grad_output->layout.shape<2>() != grad_wte->layout.shape<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Grad_output data dim not same");
  }
  auto task = TaskCompute{
      [grad_input=grad_input, grad_output=grad_output, grad_wte= grad_wte,grad_wpe=grad_wpe,
          futureDWte = *grad_wte->future, futureDWpe = *grad_wpe->future,
       futureDOutput = grad_output->future->wFuture,futureDInput = grad_input->future->wFuture](const ContextCompute *context)mutable {

        util::FutureGuard dwteRGuard{futureDWte.rFuture};
        util::FutureGuard dwteWGuard{futureDWte.wFuture};
        util::FutureGuard dwpeRGuard{futureDWpe.rFuture};
        util::FutureGuard dwpeWGuard{futureDWpe.wFuture};
        util::FutureGuard doutputGuard{futureDOutput};
        util::FutureGuard inputGuard{futureDInput};
        backwardKernel(context->cudaStream, *grad_output, *grad_input, *grad_wte, *grad_wpe);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        grad_output.reset();
        grad_input.reset();
        grad_wte.reset();
        grad_wpe.reset();
      }};
  const TaskFuture future = task.get_future();
  grad_wpe->future->wFuture  = future;
  grad_wte->future->wFuture  = future;
  grad_output->future->rFuture  = future;
  grad_input->future->rFuture = future;
  return task;
}
}  // namespace dllm::compute::GeLU
