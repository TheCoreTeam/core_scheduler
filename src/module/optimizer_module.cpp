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

#include "module/optimizer_module.h"

#include "logger.h"

namespace cs::module {
OrderedDict<std::string, std::shared_ptr<OptimizerState>>
OptimizerModule::named_states() const {
  return states_;
}

void OptimizerModule::set_lr(const double lr) const {
  for (auto kv : states_) {
    kv.value()->set_lr(lr);
  }
}

double OptimizerModule::get_lr() const {
  CS_ASSERT_TRUE(!states_.is_empty(), "Got empty state");
  return states_.begin()->value()->get_lr();
}

void OptimizerModule::zero_grad(const Scheduler& scheduler) const {}
}  // namespace cs::module

void cs::save(const module::OptimizerModule& module, const std::string& path) {
  CS_ASSERT_TRUE(false, "not implemented");
}

void cs::load(const module::OptimizerModule& module, const std::string& path) {
  CS_ASSERT_TRUE(false, "not implemented");
}
