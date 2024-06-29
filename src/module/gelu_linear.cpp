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

#include "module/gelu_linear.h"

#include "threading/scheduler.h"

namespace cs::module {
GeluLinearImpl::GeluLinearImpl(const Scheduler& scheduler,
                               const Options& options) {
  const auto state = compute::GeluLinear::init(scheduler, options);
  register_state("GeluLinearState", state);
  state_ = state;
}

Tensor GeluLinearImpl::forward(const Scheduler& scheduler,
                               const ReadOnlyTensor& input) const {
  return compute::GeluLinear::forward(scheduler, state(), input);
}

Tensor GeluLinearImpl::backward(const Scheduler& scheduler,
                                const ReadOnlyTensor& grad_output) const {
  auto grad_input =
      compute::GeluLinear::backwardInput(scheduler, state(), grad_output);
  compute::GeluLinear::backwardParameter(scheduler, state(), grad_output);
  return grad_input;
}

void GeluLinearImpl::backwardParameter(
    const Scheduler& scheduler, const ReadOnlyTensor& grad_output) const {
  compute::GeluLinear::backwardParameter(scheduler, state(), grad_output);
}

Tensor GeluLinearImpl::backwardInput(const Scheduler& scheduler,
                                     const ReadOnlyTensor& grad_output) const {
  return compute::GeluLinear::backwardInput(scheduler, state(), grad_output);
}

std::shared_ptr<compute::GeluLinear::State> GeluLinearImpl::state() const {
  return state_.lock();
}
}  // namespace cs::module
