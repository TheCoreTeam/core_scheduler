/*
 * Copyright (c) 2024 The Core Team
 *
 * Licensed under the Apache License, Version 2.0;
 * You may not use this file except in compliance with the License.
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

#include "module/amp_adamw.h"

#include "logger.h"
#include "module/amp_state.h"

namespace cs::module {
AmpAdamWImpl::AmpAdamWImpl(const Scheduler& scheduler, const Module& module,
                           const Options& options) {
  OrderedDict<std::string, std::shared_ptr<OptimizerState>> states;
  for (auto named_parameters = module.named_parameters();
       auto kv : named_parameters) {
    auto state = optimizer::AdamW::init(scheduler, kv.value(), options);
    states.insert(kv.key(), state);
  }
  states_ = states;
  module_ = module.weak_from_this();
}

void AmpAdamWImpl::step(const Scheduler& scheduler) const {
  const auto named_increments = module_.lock()->named_increments();
  for (auto states = module_.lock()->named_states(); auto& kvStates : states) {
    auto ampState =
        std::dynamic_pointer_cast<module::AmpState>(kvStates.value());
    CS_ASSERT_TRUE(ampState != nullptr, "The module is not an AMP module");
    for (auto parameters_high_precision = ampState->parameters_high_precision();
         const auto& kvP : parameters_high_precision) {
      std::string key = fmt::format("{}.{}", kvStates.key(), kvP.key());
      optimizer::AmpAdamW::step(scheduler, states_[key],
                                named_increments[key].parameter, kvP.value(),
                                named_increments[key].gradient);
    }
  }
}
}  // namespace cs::module
