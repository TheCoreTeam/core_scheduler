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
#include "compute/amp_gelu_linear.h"
#include "module/gelu_linear.h"
#include "module/pimpl.h"

namespace cs::module {
struct CS_API AmpGeluLinearImpl : GeluLinearImpl {
  using Options = compute::AmpGeluLinear::Options;

  explicit AmpGeluLinearImpl(const Scheduler &scheduler,
                             const Options &options);
};

CS_MODULE(AmpGeluLinear);
}  // namespace cs::module
