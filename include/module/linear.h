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
#include "module/module.h"
#include "module/pimpl.h"

namespace cs::module {
struct CS_API LinearImpl : Module {
  using Options = compute::Linear::Options;

  explicit LinearImpl(const Scheduler &scheduler, const Options &options);

  Tensor forward(const Scheduler &scheduler, const ReadOnlyTensor &input) const;

  Tensor backward(const Scheduler &scheduler,
                  const ReadOnlyTensor &grad_output) const;

  void backwardParameter(const Scheduler &scheduler,
                         const ReadOnlyTensor &grad_output) const;

  Tensor backwardInput(const Scheduler &scheduler,
                       const ReadOnlyTensor &grad_output) const;

  std::shared_ptr<compute::Linear::State> state() const;

 protected:
  LinearImpl() = default;

  std::weak_ptr<compute::Linear::State> state_;
};

CS_MODULE(Linear);
}  // namespace cs::module
