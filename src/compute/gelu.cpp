#include "compute/gelu.h"

#include <torch/nn/functional/activation.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"

namespace dllm::compute {
void GeLU::init(const Scheduler &scheduler, std::shared_ptr<State> &state) {
  state = std::make_shared<State>();
  scheduler.impl()->submit(TaskCompute{[](const ContextCompute *) {}});
}

void GeLU::forward(const Scheduler &scheduler,
                   const std::shared_ptr<State> &state, Tensor &output,
                   const ReadOnlyTensor &input) {
  output = Tensor{};
  auto task = TaskCompute{
      [output = output, input = input, inputFuture = utils::future(input)](
          const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::GeLU::forward");
        {
          utils::FutureGuard inputGuard{inputFuture};
          output.impl()->tensor() = torch::gelu(input.impl()->tensor());
          output.reset();
          input.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(input, future);
  utils::resetFuture(output, future);
  state->backward.input = input;
  // size
  output.sizes() = input.sizes();
  scheduler.impl()->submit(std::move(task));
}

void GeLU::backward(const Scheduler &scheduler,
                    const std::shared_ptr<State> &state, Tensor &dinput,
                    const ReadOnlyTensor &doutput) {
  dinput = Tensor{};
  auto task =
      TaskCompute{[doutput = doutput, input = state->backward.input,
                   dinput = dinput, doutputFuture = utils::future(doutput),
                   inputFuture = utils::future(state->backward.input)](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::GeLU::backward");
        {
          utils::FutureGuard doutGuard{doutputFuture};
          utils::FutureGuard inputGuard{inputFuture};
          dinput.impl()->tensor() = torch::gelu_backward(
              doutput.impl()->tensor(), input.impl()->tensor());
          dinput.reset();
          input.reset();
          doutput.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(dinput, future);
  utils::resetFuture(doutput, future);
  utils::resetFuture(state->backward.input, future);
  // size
  dinput.sizes() = doutput.sizes();
  // decrease counter
  state->backward.input.reset();
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::compute
