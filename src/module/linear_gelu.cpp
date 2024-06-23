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

#include "module/linear_gelu.h"

#include "threading/scheduler.h"

namespace cs::module {
LinearGeluImpl::LinearGeluImpl(const Scheduler& scheduler, const Options& options) {
  const auto state = compute::LinearGelu::init(scheduler, options);
  register_state("LinearState", state);
  state_ = state;
}

Tensor LinearGeluImpl::forward(const Scheduler& scheduler,
                           const ReadOnlyTensor& input) const {
  return compute::LinearGelu::forward(scheduler, state(), input);
}

Tensor LinearGeluImpl::backward(const Scheduler& scheduler,
                            const ReadOnlyTensor& grad_output) const {
  auto grad_input =
      compute::LinearGelu::backwardInput(scheduler, state(), grad_output);
  compute::LinearGelu::backwardParameter(scheduler, state(), grad_output);
  return grad_input;
}

void LinearGeluImpl::backwardParameter(const Scheduler& scheduler,
                                   const ReadOnlyTensor& grad_output) const {
  compute::LinearGelu::backwardParameter(scheduler, state(), grad_output);
}

Tensor LinearGeluImpl::backwardInput(const Scheduler& scheduler,
                                 const ReadOnlyTensor& grad_output) const {
  return compute::LinearGelu::backwardInput(scheduler, state(), grad_output);
}

std::shared_ptr<compute::LinearGelu::State> LinearGeluImpl::state() const {
  return state_.lock();
}
}  // namespace cs::module
