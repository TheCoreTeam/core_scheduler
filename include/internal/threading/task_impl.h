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
#include <latch>

#include "threading/task.h"

namespace cs {
struct Task::Impl {
  enum Type { kConfig = 0, kCompute, kLoader, kMemcpy, kNccl, kNumType };

  enum Priority { kMain = 0, kAssist, kComm, kNumPriority };

  virtual ~Impl() = default;

  Impl(std::vector<Tensor> output, std::vector<ReadOnlyTensor> input,
       const Priority priority, const Type type)
      : output_{std::move(output)},
        input_{std::move(input)},
        priority_{priority},
        type_{type} {}

  [[nodiscard]] auto &input() const { return input_; }

  [[nodiscard]] auto &output() const { return output_; }

  [[nodiscard]] auto &intermediate() const { return intermediate_; }

  [[nodiscard]] auto &type() const { return type_; }

  [[nodiscard]] auto &priority() const { return priority_; }

  virtual void operator()() const = 0;

  [[nodiscard]] virtual const char *name() const = 0;

 private:
  std::vector<Tensor> output_;

  std::vector<ReadOnlyTensor> input_;

  mutable std::vector<at::Tensor> intermediate_;

  Priority priority_;

  Type type_;
};
}  // namespace cs
