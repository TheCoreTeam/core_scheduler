#include "compute/gelu.h"

#include <torch/nn/functional/activation.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::compute {
TaskCompute GeLU::init(std::shared_ptr<State> &state) {
  state = std::make_shared<State>();
  return TaskCompute{[](const ContextCompute *) {}};
}

TaskCompute GeLU::forward(const std::shared_ptr<State> &state,
                          const std::shared_ptr<Tensor> &output,
                          const std::shared_ptr<const ReadOnlyTensor> &input) {
  auto task = TaskCompute{
      [output = output, input = input, outputfuture = output->future(),
       inputFuture =
           input->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::GeLU::forward");
        {
          util::FutureGuard outputGuard{outputfuture};
          util::FutureGuard inputGuard{inputFuture};
          DLLM_EXTRACT_TENSOR(output) = torch::gelu(DLLM_EXTRACT_TENSOR(input));
          output.reset();
          input.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  input->resetFuture(future);
  output->resetFuture(future);
  state->backward.input = input;
  // size
  output->sizes() = input->sizes();
  return task;
}

TaskCompute GeLU::backward(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &dinput,
    const std::shared_ptr<const ReadOnlyTensor> &doutput) {
  auto task =
      TaskCompute{[doutput = doutput, input = state->backward.input,
                   dinput = dinput, dinputFuture = dinput->future(),
                   doutputFuture = doutput->future(),
                   inputFuture = state->backward.input->future()](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::GeLU::backward");
        {
          util::FutureGuard dinputGuard{dinputFuture};
          util::FutureGuard doutGuard{doutputFuture};
          util::FutureGuard inputGuard{inputFuture};
          DLLM_EXTRACT_TENSOR(dinput) = torch::gelu_backward(
              DLLM_EXTRACT_TENSOR(doutput), DLLM_EXTRACT_TENSOR(input));
          dinput.reset();
          input.reset();
          doutput.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  dinput->resetFuture(future);
  doutput->resetFuture(future);
  state->backward.input->resetFuture(future);
  // size
  dinput->sizes() = doutput->sizes();
  // decrease counter
  state->backward.input.reset();
  return task;
}
}  // namespace dllm::compute
