/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "compute/cross_entropy.h"

#include <ATen/TensorOperators.h>
#include <ATen/ops/log_softmax.h>

#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute {
std::shared_ptr<CrossEntropy::State> CrossEntropy::init(
    const Scheduler &scheduler, const Options &options) {
  CS_ASSERT_TRUE(options.label_smoothing() == 0.0,
                 "We do not support label_smoothing");
  return std::make_shared<State>(
      State::Forward{}, State::Backward{},
      State::Args{options.reduction(), options.ignore_index(),
                  options.label_smoothing()});
}

Tensor CrossEntropy::forward(const Scheduler &scheduler,
                             const std::shared_ptr<State> &state,
                             const ReadOnlyTensor &input,
                             const ReadOnlyTensor &target) {
  struct Impl : Task::Impl {
    State::Args args;

    explicit Impl(
        std::vector<Tensor> output /* log_probs, loss, total_weight */,
        std::vector<ReadOnlyTensor> input /* weight, input, target */,
        const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          args{args} {}
    void operator()() const override {
      const c10::optional weight_{
          !input()[0].impl()->tensor().defined()
              ? c10::optional<at::Tensor>{}
              : c10::optional{input()[0].impl()->tensor()}};
      // TODO(Jie): maybe change dtype for amp?
      output()[0].impl()->tensor() =
          at::log_softmax(input()[1].impl()->tensor(), 1);
      std::make_tuple(std::ref(output()[1].impl()->tensor()),
                      std::ref(output()[2].impl()->tensor())) =
          at::nll_loss_forward(output()[0].impl()->tensor(),
                               input()[2].impl()->tensor(), weight_,
                               args.reduction, args.ignore_index);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::CrossEntropy::forward";
    }
  };

  Tensor loss{};
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
  return loss;
}

Tensor CrossEntropy::backward(const Scheduler &scheduler,
                              const std::shared_ptr<State> &state) {
  struct Impl : Task::Impl {
    State::Args args;

    explicit Impl(std::vector<Tensor> output /* grad_input */,
                  std::vector<ReadOnlyTensor>
                      input /* weight, loss, log_probs, target, total_weight */,
                  const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          args{args} {}
    void operator()() const override {
      const c10::optional weight_{
          !input()[0].impl()->tensor().defined()
              ? c10::optional<at::Tensor>{}
              : c10::optional{input()[0].impl()->tensor()}};

      // TODO(Jie): maybe change dtype for amp?
      const auto dnll = at::nll_loss_backward(
          at::ones_like(input()[1].impl()->tensor()),
          input()[2].impl()->tensor(), input()[3].impl()->tensor(), weight_,
          args.reduction, args.ignore_index, input()[4].impl()->tensor());
      output()[0].impl()->tensor() = at::_log_softmax_backward_data(
          dnll, input()[2].impl()->tensor(), 1,
          input()[2].impl()->tensor().scalar_type());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::CrossEntropy::backward";
    }
  };

  Tensor grad_input{};
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
  return grad_input;
}
}  // namespace cs::compute
