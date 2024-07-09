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

#include "module/adamw.h"

namespace cs::module {
AdamWImpl::AdamWImpl(const Scheduler& scheduler, const Module& module,
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

void AdamWImpl::step(const Scheduler& scheduler) const {
  auto named_increments = module_.lock()->named_increments();
  for (const auto& kv : states_) {
    auto& increment = named_increments[kv.key()];
    optimizer::AdamW::step(scheduler, kv.value(), increment.parameter,
                           increment.gradient);
  }
}
}  // namespace cs::module
