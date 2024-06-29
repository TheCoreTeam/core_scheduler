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
struct Linear {
  struct State : virtual module::State {
    struct Forward {
      Tensor weight;
      Tensor bias{};
      Tensor grad_weight{};
      Tensor grad_bias{};
      std::shared_ptr<module::OptimizerState> optimizer_weight = nullptr;
      std::shared_ptr<module::OptimizerState> optimizer_bias = nullptr;
    } forward;
    struct Backward {
      ReadOnlyTensor input;
    } backward;
    struct Args {
      const bool bias;
    } args;

    State(const Forward &forward, const Backward &backward, const Args &args)
        : forward{forward}, backward{backward}, args{args} {}

    [[nodiscard]] OrderedDict<std::string, Tensor> parameters() const override;

    [[nodiscard]] OrderedDict<std::string, Increment> increments() override;
  };

  struct Options {
    Options(const int64_t in_futures, const int64_t out_futures)
        : in_futures_(in_futures), out_futures_(out_futures) {}
    CS_ARG(int64_t, in_futures);
    CS_ARG(int64_t, out_futures);
    CS_ARG(bool, bias) = true;
    CS_ARG(c10::optional<at::Device>, device) = {};
    CS_ARG(c10::optional<at::ScalarType>, dtype) = {};
  };

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options);

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &input);

  static Tensor backwardInput(const Scheduler &scheduler,
                              const std::shared_ptr<State> &state,
                              const ReadOnlyTensor &grad_output);

  static void backwardParameter(const Scheduler &scheduler,
                                const std::shared_ptr<State> &state,
                                const ReadOnlyTensor &grad_output);
};
}  // namespace cs::compute
