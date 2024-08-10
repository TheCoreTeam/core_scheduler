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
#include <torch/nn/pimpl.h>
#include <torch/ordered_dict.h>

#include <memory>
#include <string>

#include "module/module.h"
#include "module/state.h"
#include "threading/scheduler.h"

namespace cs::module {
struct CS_API OptimizerModule : std::enable_shared_from_this<OptimizerModule> {
  virtual ~OptimizerModule() = default;

  OrderedDict<std::string, std::shared_ptr<OptimizerState>> named_states()
      const;

  void set_lr(double lr) const;

  double get_lr() const;

  virtual void step(const Scheduler& scheduler) const = 0;

  // This is set_to_none by default!
  void zero_grad(const Scheduler& scheduler) const;

  void to(TensorOptions options) const;

 protected:
  OrderedDict<std::string, std::shared_ptr<OptimizerState>> states_;

  std::weak_ptr<const Module> module_;
};
}  // namespace cs::module

namespace cs {
CS_API void save(const module::OptimizerModule& module,
                 const std::string& path);

CS_API void load(const module::OptimizerModule& module,
                 const std::string& path);
}  // namespace cs
