#include "compute/cross_entropy.h"

#include <ATen/TensorOperators.h>
#include <ATen/ops/log_softmax.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::compute {
TaskCompute CrossEntropy::init(std::shared_ptr<State> &state,
                               const Options &options) {
  DLLM_ASSERT_TRUE(options.label_smoothing() == 0.0,
                   "We do not support label_smoothing");
  state = std::make_shared<State>(
      State::Forward{}, State::Backward{},
      State::Args{options.reduction(), options.ignore_index(),
                  options.label_smoothing()});
  return TaskCompute{[](const ContextCompute *) {}};
}

TaskCompute CrossEntropy::forward(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &loss,
    const std::shared_ptr<const ReadOnlyTensor> &input,
    const std::shared_ptr<const ReadOnlyTensor> &target) {
  auto log_probs = Tensor::create();
  auto total_weight = Tensor::create();
  auto task = TaskCompute{
      [weight = state->forward.weight, reduction = state->args.reduction,
       ignore_index = state->args.ignore_index,
       label_smoothing = state->args.label_smoothing, loss = loss,
       input = input, target = target, log_probs = log_probs,
       total_weight = total_weight, lossfuture = loss->future(),
       weightFuture = state->forward.weight->future(),
       inputFuture = input->future(),
       targetFuture = target->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::CrossEntropy::forward");
        {
          util::FutureGuard lossGuard{lossfuture};
          util::FutureGuard inputGuard{inputFuture};
          util::FutureGuard weightGuard{weightFuture};
          util::FutureGuard targetGuard{targetFuture};
          const c10::optional weight_{
              !DLLM_EXTRACT_TENSOR(weight).defined()
                  ? c10::optional<at::Tensor>{}
                  : c10::optional{DLLM_EXTRACT_TENSOR(weight)}};
          DLLM_EXTRACT_TENSOR(log_probs) =
              at::log_softmax(DLLM_EXTRACT_TENSOR(input), 1);
          std::make_tuple(std::ref(DLLM_EXTRACT_TENSOR(loss)),
                          std::ref(DLLM_EXTRACT_TENSOR(total_weight))) =
              at::nll_loss_forward(DLLM_EXTRACT_TENSOR(log_probs),
                                   DLLM_EXTRACT_TENSOR(target), weight_,
                                   reduction, ignore_index);

          loss.reset();
          log_probs.reset();
          total_weight.reset();
          input.reset();
          weight.reset();
          target.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  loss->resetFuture(future);
  input->resetFuture(future);
  target->resetFuture(future);
  log_probs->resetFuture(future);
  total_weight->resetFuture(future);
  state->forward.weight->resetFuture(future);
  state->backward.log_probs = log_probs;
  state->backward.total_weight = std::move(total_weight);
  state->backward.target = target;
  state->backward.loss = loss;
  // size
  log_probs->sizes() = input->sizes();
  loss->sizes() = IntArray{1};
  return task;
}

TaskCompute CrossEntropy::backward(const std::shared_ptr<State> &state,
                                   const std::shared_ptr<Tensor> &dinput) {
  auto task = TaskCompute{
      [weight = state->forward.weight, log_probs = state->backward.log_probs,
       total_weight = state->backward.total_weight,
       target = state->backward.target, loss = state->backward.loss,
       reduction = state->args.reduction,
       ignore_index = state->args.ignore_index, dinput = dinput,
       dinputFuture = dinput->future(),
       weightFuture = state->forward.weight->future(),
       log_probsFuture = state->backward.log_probs->future(),
       total_weightFuture = state->backward.total_weight->future(),
       targetFuture = state->backward.target->future(),
       lossFuture = state->backward.loss->future()](
          const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::CrossEntropy::backward");
        {
          util::FutureGuard dinputGuard{dinputFuture};
          util::FutureGuard log_probsGuard{log_probsFuture};
          util::FutureGuard total_weightGuard{total_weightFuture};
          util::FutureGuard targetGuard{targetFuture};
          util::FutureGuard lossGuard{lossFuture};
          util::FutureGuard weightGuard{weightFuture};

          const c10::optional weight_{
              !DLLM_EXTRACT_TENSOR(weight).defined()
                  ? c10::optional<at::Tensor>{}
                  : c10::optional{DLLM_EXTRACT_TENSOR(weight)}};

          const auto dnll = at::nll_loss_backward(
              at::ones_like(DLLM_EXTRACT_TENSOR(loss)),
              DLLM_EXTRACT_TENSOR(log_probs), DLLM_EXTRACT_TENSOR(target),
              weight_, reduction, ignore_index,
              DLLM_EXTRACT_TENSOR(total_weight));
          DLLM_EXTRACT_TENSOR(dinput) = at::_log_softmax_backward_data(
              dnll, DLLM_EXTRACT_TENSOR(log_probs), 1,
              DLLM_EXTRACT_TENSOR(log_probs).scalar_type());

          dinput.reset();
          log_probs.reset();
          total_weight.reset();
          target.reset();
          loss.reset();
          weight.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  dinput->resetFuture(future);
  state->forward.weight->resetFuture(future);
  state->backward.log_probs->resetFuture(future);
  state->backward.total_weight->resetFuture(future);
  state->backward.target->resetFuture(future);
  state->backward.loss->resetFuture(future);
  // size
  dinput->sizes() = state->backward.log_probs->sizes();
  // decrease counter
  state->backward.log_probs.reset();
  state->backward.total_weight.reset();
  state->backward.target.reset();
  state->backward.loss.reset();
  return task;
}
}  // namespace dllm::compute
