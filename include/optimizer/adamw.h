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
#include "module/module.h"
#include "module/state.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace cs::module {
struct Module;
}

namespace cs::optimizer {
struct CS_API AdamW {
  AdamW() = delete;

  struct State final : module::OptimizerState {
    struct Tensors {
      Tensor m;
      Tensor v;
      Tensor v_max{};
    } tensors;
    struct Options {
      mutable double lr = 1e-3;
      const double beta1 = 0.9;
      const double beta2 = 0.999;
      const double eps = 1e-8;
      const double weight_decay = 1e-2;
      const bool amsgrad = false;
      long t = 0;
    } options;

    State(const Tensors &tensors, const Options &options);

    void set_lr(double lr) const override;

    double get_lr() const override;
  };

  struct Options {
    CS_ARG(double, lr) = 1e-3;
    CS_ARG(double, beta1) = 0.9;
    CS_ARG(double, beta2) = 0.999;
    CS_ARG(double, eps) = 1e-8;
    CS_ARG(double, weight_decay) = 1e-2;
    CS_ARG(bool, amsgrad) = false;
    CS_ARG(long, t) = 0;
  };

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const ReadOnlyTensor &parameter,
                                     const Options &options);

  static void step(const Scheduler &scheduler,
                   const std::shared_ptr<module::OptimizerState> &state,
                   const Tensor &w, const ReadOnlyTensor &dw);
};
}  // namespace cs::optimizer
