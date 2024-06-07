#include "compute/cross_entropy.h"

#include <ATen/TensorOperators.h>
#include <ATen/ops/log_softmax.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"

namespace dllm::compute {
void CrossEntropy::init(const Scheduler &scheduler,
                        std::shared_ptr<State> &state, const Options &options) {
  DLLM_ASSERT_TRUE(options.label_smoothing() == 0.0,
                   "We do not support label_smoothing");
  state = std::make_shared<State>(
      State::Forward{}, State::Backward{},
      State::Args{options.reduction(), options.ignore_index(),
                  options.label_smoothing()});
}

void CrossEntropy::forward(const Scheduler &scheduler,
                           const std::shared_ptr<State> &state, Tensor &loss,
                           const ReadOnlyTensor &input,
                           const ReadOnlyTensor &target) {
  loss = Tensor{};
  Tensor log_probs;
  Tensor total_weight;
  auto task = TaskCompute{[weight = state->forward.weight,
                           reduction = state->args.reduction,
                           ignore_index = state->args.ignore_index,
                           label_smoothing = state->args.label_smoothing,
                           loss = loss, input = input, target = target,
                           log_probs = log_probs, total_weight = total_weight,
                           weightFuture = utils::future(state->forward.weight),
                           inputFuture = utils::future(input),
                           targetFuture = utils::future(target)](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::CrossEntropy::forward");
    {
      utils::FutureGuard inputGuard{inputFuture};
      utils::FutureGuard weightGuard{weightFuture};
      utils::FutureGuard targetGuard{targetFuture};
      const c10::optional weight_{!weight.impl()->tensor().defined()
                                      ? c10::optional<at::Tensor>{}
                                      : c10::optional{weight.impl()->tensor()}};
      log_probs.impl()->tensor() = at::log_softmax(input.impl()->tensor(), 1);
      std::make_tuple(std::ref(loss.impl()->tensor()),
                      std::ref(total_weight.impl()->tensor())) =
          at::nll_loss_forward(log_probs.impl()->tensor(),
                               target.impl()->tensor(), weight_, reduction,
                               ignore_index);

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
  utils::resetFuture(loss, future);
  utils::resetFuture(input, future);
  utils::resetFuture(target, future);
  utils::resetFuture(log_probs, future);
  utils::resetFuture(total_weight, future);
  utils::resetFuture(state->forward.weight, future);
  state->backward.log_probs = log_probs;
  state->backward.total_weight = std::move(total_weight);
  state->backward.target = target;
  state->backward.loss = loss;
  // size
  log_probs.sizes() = input.sizes();
  loss.sizes() = IntArray{1};
  scheduler.impl()->submit(std::move(task));
}

void CrossEntropy::backward(const Scheduler &scheduler,
                            const std::shared_ptr<State> &state,
                            Tensor &dinput) {
  dinput = Tensor{};
  auto task = TaskCompute{
      [weight = state->forward.weight, log_probs = state->backward.log_probs,
       total_weight = state->backward.total_weight,
       target = state->backward.target, loss = state->backward.loss,
       reduction = state->args.reduction,
       ignore_index = state->args.ignore_index, dinput = dinput,
       weightFuture = utils::future(state->forward.weight),
       log_probsFuture = utils::future(state->backward.log_probs),
       total_weightFuture = utils::future(state->backward.total_weight),
       targetFuture = utils::future(state->backward.target),
       lossFuture = utils::future(state->backward.loss)](
          const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::CrossEntropy::backward");
        {
          utils::FutureGuard log_probsGuard{log_probsFuture};
          utils::FutureGuard total_weightGuard{total_weightFuture};
          utils::FutureGuard targetGuard{targetFuture};
          utils::FutureGuard lossGuard{lossFuture};
          utils::FutureGuard weightGuard{weightFuture};

          const c10::optional weight_{
              !weight.impl()->tensor().defined()
                  ? c10::optional<at::Tensor>{}
                  : c10::optional{weight.impl()->tensor()}};

          const auto dnll = at::nll_loss_backward(
              at::ones_like(loss.impl()->tensor()), log_probs.impl()->tensor(),
              target.impl()->tensor(), weight_, reduction, ignore_index,
              total_weight.impl()->tensor());
          dinput.impl()->tensor() = at::_log_softmax_backward_data(
              dnll, log_probs.impl()->tensor(), 1,
              log_probs.impl()->tensor().scalar_type());

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
  utils::resetFuture(dinput, future);
  utils::resetFuture(state->forward.weight, future);
  utils::resetFuture(state->backward.log_probs, future);
  utils::resetFuture(state->backward.total_weight, future);
  utils::resetFuture(state->backward.target, future);
  utils::resetFuture(state->backward.loss, future);
  // size
  dinput.sizes() = state->backward.log_probs.sizes();
  // decrease counter
  state->backward.log_probs.reset();
  state->backward.total_weight.reset();
  state->backward.target.reset();
  state->backward.loss.reset();
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::compute
