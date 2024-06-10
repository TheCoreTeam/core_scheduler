#include "compute/cross_entropy.h"

#include <ATen/TensorOperators.h>
#include <ATen/ops/log_softmax.h>

#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

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
  struct Impl : Task::Impl {
    State::Args args;

    explicit Impl(
        std::vector<Tensor> output /* log_probs, loss, total_weight */,
        std::vector<ReadOnlyTensor> input /* weight, input, target */,
        const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), compute},
          args{args} {}
    void operator()() const override {
      const c10::optional weight_{
          !input()[0].impl()->tensor().defined()
              ? c10::optional<at::Tensor>{}
              : c10::optional{input()[0].impl()->tensor()}};
      output()[0].impl()->tensor() =
          at::log_softmax(input()[1].impl()->tensor(), 1);
      std::make_tuple(std::ref(output()[1].impl()->tensor()),
                      std::ref(output()[2].impl()->tensor())) =
          at::nll_loss_forward(output()[0].impl()->tensor(),
                               input()[2].impl()->tensor(), weight_,
                               args.reduction, args.ignore_index);
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::CrossEntropy::forward";
    }
  };

  loss = Tensor{};
  Tensor log_probs;
  Tensor total_weight;
  state->backward.log_probs = log_probs;
  state->backward.total_weight = total_weight;
  state->backward.target = target;
  state->backward.loss = loss;
  // size
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{log_probs, loss, total_weight},
                                       {state->forward.weight, input, target},
                                       state->args})});
}

void CrossEntropy::backward(const Scheduler &scheduler,
                            const std::shared_ptr<State> &state,
                            Tensor &grad_input) {
  struct Impl : Task::Impl {
    State::Args args;

    explicit Impl(std::vector<Tensor> output /* grad_input */,
                  std::vector<ReadOnlyTensor>
                      input /* weight, loss, log_probs, target, total_weight */,
                  const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), compute},
          args{args} {}
    void operator()() const override {
      const c10::optional weight_{
          !input()[0].impl()->tensor().defined()
              ? c10::optional<at::Tensor>{}
              : c10::optional{input()[0].impl()->tensor()}};

      const auto dnll = at::nll_loss_backward(
          at::ones_like(input()[1].impl()->tensor()),
          input()[2].impl()->tensor(), input()[3].impl()->tensor(), weight_,
          args.reduction, args.ignore_index, input()[4].impl()->tensor());
      output()[0].impl()->tensor() = at::_log_softmax_backward_data(
          dnll, input()[2].impl()->tensor(), 1,
          input()[2].impl()->tensor().scalar_type());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::CrossEntropy::backward";
    }
  };

  grad_input = Tensor{};
  // size
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{
      {grad_input},
      {state->forward.weight, state->backward.loss, state->backward.log_probs,
       state->backward.target, state->backward.total_weight},
      state->args})});
  // decrease counter
  state->backward.log_probs.reset();
  state->backward.total_weight.reset();
  state->backward.target.reset();
  state->backward.loss.reset();
}
}  // namespace dllm::compute
