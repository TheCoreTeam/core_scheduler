#include "compute/cross_entropy.h"

#include <ATen/ops/cross_entropy_loss.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/log_softmax.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::compute {
TaskCompute CrossEntropy::init(
    std::shared_ptr<State> &state,
    const std::shared_ptr<const ReadOnlyTensor> &weight,
    const int64_t reduction, const int64_t ignore_index,
    const double label_smoothing) {
  DLLM_ASSERT_TRUE(weight == nullptr, "We do not support weight");
  DLLM_ASSERT_TRUE(ignore_index == -100, "We do not support ignore_index");
  DLLM_ASSERT_TRUE(label_smoothing == 0.0, "We do not support label_smoothing");
  state = std::make_shared<State>(
      State::Forward{}, State::Backward{},
      State::Args{weight, reduction, ignore_index, label_smoothing});
  return TaskCompute{[](const ContextCompute *) {}};
}

TaskCompute CrossEntropy::forward(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &loss,
    const std::shared_ptr<const ReadOnlyTensor> &input,
    const std::shared_ptr<const ReadOnlyTensor> &target) {
  auto task = TaskCompute{
      [weight = state->args.weight, reduction = state->args.reduction,
       ignore_index = state->args.ignore_index,
       label_smoothing = state->args.label_smoothing, loss = loss,
       input = input, target = target, lossfuture = loss->future(),
       inputFuture = input->future(),
       targetFuture = target->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::CrossEntropy::forward");
        {
          util::FutureGuard lossGuard{lossfuture};
          util::FutureGuard inputGuard{inputFuture};
          util::FutureGuard targetGuard{targetFuture};
          const c10::optional weight_{
              weight == nullptr ? c10::optional<at::Tensor>{nullptr}
                                : c10::optional{DLLM_EXTRACT_TENSOR(weight)}};
          DLLM_EXTRACT_TENSOR(loss) = at::cross_entropy_loss(
              DLLM_EXTRACT_TENSOR(input), DLLM_EXTRACT_TENSOR(target), weight_,
              reduction, ignore_index, label_smoothing);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  loss->resetFuture(future);
  input->resetFuture(future);
  target->resetFuture(future);
  state->backward.input = input;
  state->backward.target = target;
  // size
  loss->sizes() = IntArray{input->size(0)};
  return task;
}

TaskCompute CrossEntropy::backward(const std::shared_ptr<State> &state,
                                   const std::shared_ptr<Tensor> &dinput) {
  auto task = TaskCompute{[input = state->backward.input,
                           target = state->backward.target,
                           reduction = state->args.reduction, dinput = dinput,
                           dinputFuture = dinput->future(),
                           inputFuture = state->backward.input->future(),
                           targetFuture = state->backward.target->future()](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::CrossEntropy::backward");
    {
      util::FutureGuard dinputGuard{dinputFuture};
      util::FutureGuard inputGuard{inputFuture};
      util::FutureGuard targetGuard{targetFuture};

      const auto log_softmax = at::log_softmax(DLLM_EXTRACT_TENSOR(input), 1);
      auto grad = at::exp(log_softmax);
      grad.scatter_(1, DLLM_EXTRACT_TENSOR(target).unsqueeze(1), -1.0);

      if (reduction == at::Reduction::Mean) {
        grad /= DLLM_EXTRACT_TENSOR(input).size(0);
      }
      DLLM_EXTRACT_TENSOR(dinput) = grad;

      dinput.reset();
      input.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  dinput->resetFuture(future);
  state->backward.input->resetFuture(future);
  state->backward.target->resetFuture(future);
  // size
  dinput->sizes() = state->backward.input->sizes();
  // decrease counter
  state->backward.input.reset();
  state->backward.target.reset();
  return task;
}
}  // namespace dllm::compute
