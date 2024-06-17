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

#include "arg.h"
#include "module/state.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::module {
struct Module;
}

namespace dllm {
struct ThreadPoolCompute;
}

namespace dllm::optimizer {
struct AdamW {
  AdamW() = delete;

  struct State final : module::OptimizerState {
    struct Tensors {
      Tensor m;
      Tensor v;
      Tensor vMax{};
    } tensors;
    struct Options {
      const double lr = 1e-3;
      const double beta1 = 0.9;
      const double beta2 = 0.999;
      const double eps = 1e-8;
      const double weight_decay = 1e-2;
      const bool amsgrad = false;
      long t = 0;
    } options;

    State(const Tensors &tensors, const Options &options)
        : tensors{tensors}, options{options} {}
  };

  struct Options {
    DLLM_ARG(double, lr) = 1e-3;
    DLLM_ARG(double, beta1) = 0.9;
    DLLM_ARG(double, beta2) = 0.999;
    DLLM_ARG(double, eps) = 1e-8;
    DLLM_ARG(double, weight_decay) = 1e-2;
    DLLM_ARG(bool, amsgrad) = false;
    DLLM_ARG(long, t) = 0;
  };

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
                   const ReadOnlyTensor &dw);
};
}  // namespace dllm::optimizer
