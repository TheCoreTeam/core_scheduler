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
#include <memory>
#include <vector>

#include "tensor.h"

namespace cs {
struct Task {
  struct Impl;

  CS_API explicit Task(std::shared_ptr<Impl> impl);

  CS_API [[nodiscard]] const std::vector<ReadOnlyTensor> &input() const;

  CS_API [[nodiscard]] const std::vector<Tensor> &output() const;

  CS_API [[nodiscard]] const char *name() const;

  CS_API void operator()() const;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

  CS_API void reset();

 private:
  std::shared_ptr<Impl> impl_;
};
}  // namespace cs
