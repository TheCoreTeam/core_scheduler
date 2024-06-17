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

#include "module/linear.h"

#include "threading/scheduler.h"

namespace cs::module {
LinearImpl::LinearImpl(const Scheduler& scheduler, const Options& options) {
  const auto state = compute::Linear::init(scheduler, options);
  register_state("LinearState", state);
  state_ = state;
}

Tensor LinearImpl::forward(const Scheduler& scheduler,
                           const ReadOnlyTensor& input) const {
  return compute::Linear::forward(scheduler, state(), input);
}

Tensor LinearImpl::backward(const Scheduler& scheduler,
                            const ReadOnlyTensor& grad_output) const {
  auto grad_input =
      compute::Linear::backwardInput(scheduler, state(), grad_output);
  compute::Linear::backwardParameter(scheduler, state(), grad_output);
  return grad_input;
}

void LinearImpl::backwardParameter(const Scheduler& scheduler,
                                   const ReadOnlyTensor& grad_output) const {
  compute::Linear::backwardParameter(scheduler, state(), grad_output);
}

Tensor LinearImpl::backwardInput(const Scheduler& scheduler,
                                 const ReadOnlyTensor& grad_output) const {
  return compute::Linear::backwardInput(scheduler, state(), grad_output);
}

std::shared_ptr<compute::Linear::State> LinearImpl::state() const {
  return state_.lock();
}
}  // namespace cs::module
