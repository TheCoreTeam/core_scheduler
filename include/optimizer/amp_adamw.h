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
#include <type_traits>

#include "optimizer/adamw.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace cs::module {
struct Module;
}

namespace cs::optimizer {
struct CS_API AmpAdamW : AdamW {
  AmpAdamW() = delete;

  using State = AdamW::State;
  using Options = AdamW::Options;

  static void init(const Scheduler &scheduler, const module::Module &module,
                   const Options &options);

  template <typename Module, typename = std::enable_if_t<
                                 !std::is_base_of_v<module::Module, Module> &&
                                 !std::is_base_of_v<ReadOnlyTensor, Module>>>
  static void init(const Scheduler &scheduler, const Module &module,
                   const Options &options) {
    init(scheduler, *module, options);
  }

  static void step(const Scheduler &scheduler, const module::Module &module);

  template <typename Module, typename = std::enable_if_t<
                                 !std::is_base_of_v<module::Module, Module>>>
  static void step(const Scheduler &scheduler, const Module &module) {
    step(scheduler, *module);
  }

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const ReadOnlyTensor &parameter,
                                     const Options &options);

  static void step(const Scheduler &scheduler,
                   const std::shared_ptr<State> &state, Tensor &w,
                   Tensor &wFp32, const ReadOnlyTensor &dw);
};
}  // namespace cs::optimizer
