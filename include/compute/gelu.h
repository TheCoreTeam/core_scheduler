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
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::compute {
struct GeLU {
  struct State {
    struct Forward {
    } forward;
    struct Backward {
      ReadOnlyTensor input;
    } backward;
    struct Args {
    } args;
  };

  static std::shared_ptr<State> init(const Scheduler &scheduler);

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &input);

  static Tensor backward(const Scheduler &scheduler,
                         const std::shared_ptr<State> &state,
                         const ReadOnlyTensor &grad_output);
};
}  // namespace dllm::compute
