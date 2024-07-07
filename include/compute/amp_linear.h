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
#include "compute/linear.h"
#include "module/amp_state.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace cs::compute {
struct CS_API AmpLinear {
  struct State final : module::AmpState, Linear::State {
    using Forward = Linear::State::Forward;
    using Backward = Linear::State::Backward;
    using Args = Linear::State::Args;

    struct ForwardHighPrecision {
      Tensor weight{};
      Tensor bias{};
    } forward_high_precision;

    State(const Forward &forward,
          const ForwardHighPrecision &forwardHighPrecision,
          const Backward &backward, const Args &args);

    [[nodiscard]] OrderedDict<std::string, Tensor> parameters_high_precision()
        const override;

    [[nodiscard]] OrderedDict<std::string, Tensor> parameters() const override;
  };

  using Options = Linear::Options;

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options);
};
}  // namespace cs::compute
