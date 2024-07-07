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
#include "module/optimizer_module.h"
#include "module/pimpl.h"
#include "optimizer/amp_adamw.h"

namespace cs::module {
struct CS_API AmpAdamWImpl : OptimizerModule {
  using Options = optimizer::AmpAdamW::Options;

  explicit AmpAdamWImpl(const Scheduler &scheduler, const Module &module,
                        const Options &options);

  template <typename Module, typename = std::enable_if_t<
                                 !std::is_base_of_v<module::Module, Module>>>
  explicit AmpAdamWImpl(const Scheduler &scheduler, const Module &module,
                        const Options &options)
      : AmpAdamWImpl{scheduler, *module, options} {}

  void step(const Scheduler &scheduler) const override;

 protected:
  AmpAdamWImpl() = default;
};

CS_MODULE(AmpAdamW);
}  // namespace cs::module
