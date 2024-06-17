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

#include "module/layer_norm.h"

#include "threading/scheduler.h"

namespace cs::module {
LayerNormImpl::LayerNormImpl(const Scheduler& scheduler,
                             const Options& options) {
  const auto state = compute::LayerNorm::init(scheduler, options);
  register_state("LayerNormState", state);
  state_ = state;
}

Tensor LayerNormImpl::forward(const Scheduler& scheduler,
                              const ReadOnlyTensor& input) const {
  return compute::LayerNorm::forward(scheduler, state(), input);
}

Tensor LayerNormImpl::backward(const Scheduler& scheduler,
                               const ReadOnlyTensor& grad_output) const {
  return compute::LayerNorm::backward(scheduler, state(), grad_output);
}

std::shared_ptr<compute::LayerNorm::State> LayerNormImpl::state() const {
  return state_.lock();
}
}  // namespace cs::module
