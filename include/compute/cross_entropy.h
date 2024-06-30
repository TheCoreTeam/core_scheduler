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
#include <ATen/core/Reduction.h>

#include "arg.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace cs::compute {
struct CS_API CrossEntropy {
  struct State {
    struct Forward {
      ReadOnlyTensor weight;
    } forward;
    struct Backward {
      ReadOnlyTensor total_weight;
      ReadOnlyTensor log_probs;
      ReadOnlyTensor target;
      ReadOnlyTensor loss;
    } backward;
    struct Args {
      int64_t reduction;
      int64_t ignore_index;
      double label_smoothing;
    } args;
  };

  struct Options {
    Options() {}
    CS_ARG(at::Reduction::Reduction, reduction) = at::Reduction::Mean;
    CS_ARG(int64_t, ignore_index) = -100;
    CS_ARG(double, label_smoothing) = 0.0;
  };

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options = {});

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &input,
                        const ReadOnlyTensor &target);

  static Tensor backward(const Scheduler &scheduler,
                         const std::shared_ptr<State> &state);
};
}  // namespace cs::compute
