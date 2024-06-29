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
#include "compute/linear.h"
#include "module/amp_state.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace cs::compute {
struct AmpLinear {
  struct State final : module::AmpState, Linear::State {
    using Forward = Linear::State::Forward;
    using Backward = Linear::State::Backward;
    using Args = Linear::State::Args;

    struct ForwardFp32 {
      Tensor weight{};
      Tensor bias{};
    } forwardFp32;

    State(const Forward &forward, const ForwardFp32 &forwardFp32,
          const Backward &backward, const Args &args);

    [[nodiscard]] OrderedDict<std::string, Tensor> parametersFp32()
        const override;
  };

  using Options = Linear::Options;

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options);
};
}  // namespace cs::compute
