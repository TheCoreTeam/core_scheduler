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

#pragma once
#include "arg.h"
#include "module/state.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace cs::compute {
struct CS_API LayerNorm {
  struct State : virtual module::State {
    struct Forward {
      Tensor weight;
      Tensor bias{};
      Tensor grad_weight{};
      Tensor grad_bias{};
    } forward;
    struct Backward {
      ReadOnlyTensor input{};
      ReadOnlyTensor mean{};
      ReadOnlyTensor rstd{};
    } backward;
    struct Args {
      const IntArray normalized_shape;
      const double eps;
      const bool elementwise_affine;
      const bool bias;
    } args;

    State(const Forward &forward, const Backward &backward, const Args &args);

    [[nodiscard]] OrderedDict<std::string, Tensor> parameters() const override;

    [[nodiscard]] OrderedDict<std::string, Tensor> gradients() const override;

    [[nodiscard]] OrderedDict<std::string, Increment> increments()
        const override;

    void zero_grad() override;
  };

  struct Options {
    Options(IntArrayRef normalized_shape)
        : normalized_shape_{normalized_shape} {}
    CS_ARG(IntArray, normalized_shape);
    CS_ARG(double, eps) = 1e-05;
    CS_ARG(bool, elementwise_affine) = true;
    CS_ARG(bool, bias) = true;
    CS_ARG(c10::optional<at::Device>, device) = {};
    CS_ARG(c10::optional<at::ScalarType>, dtype) = {};
  };

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options);

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &input);

  static Tensor backward(const Scheduler &scheduler,
                         const std::shared_ptr<State> &state,
                         const ReadOnlyTensor &grad_output);

private:
  static Tensor backward_input(const Scheduler &scheduler,
                               const std::shared_ptr<State> &state,
                               const ReadOnlyTensor &grad_output);

  static void backward_parameter(const Scheduler &scheduler,
                                   const std::shared_ptr<State> &state,
                                   const ReadOnlyTensor &grad_output);
};
}  // namespace cs::compute
