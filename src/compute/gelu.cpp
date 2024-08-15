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

#include "compute/gelu.h"

#include <ATen/ops/gelu.h>
#include <ATen/ops/gelu_backward.h>

#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute {
std::shared_ptr<GeLU::State> GeLU::init(const Scheduler &scheduler) {
  return std::make_shared<State>();
}

Tensor GeLU::forward(const Scheduler &scheduler,
                     const std::shared_ptr<State> &state,
                     const ReadOnlyTensor &input) {
  struct Impl : Task::Impl {
    Impl(std::vector<Tensor> output /* output */,
         std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::gelu(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::GeLU::forward";
    }
  };

  Tensor output{};
  state->backward.input = input;
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output}, {input}})});
  return output;
}

Tensor GeLU::backward(const Scheduler &scheduler,
                      const std::shared_ptr<State> &state,
                      const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    Impl(std::vector<Tensor> output /* grad_input */,
         std::vector<ReadOnlyTensor> input /* grad_ouput, input */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::gelu_backward(
          input()[0].impl()->tensor(), input()[1].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::GeLU::backward";
    }
  };

  Tensor grad_input{};
  // decrease counter
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{grad_input}, {grad_output, state->backward.input}})});
  state->backward.input.reset();
  return grad_input;
}
}  // namespace cs::compute
