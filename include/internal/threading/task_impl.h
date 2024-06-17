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

namespace dllm {
struct Task::Impl {
  enum Type {
    compute,
    loader,
    memcpy,
    nccl,
  };

  virtual ~Impl() = default;

  Impl(std::vector<Tensor> output, std::vector<ReadOnlyTensor> input,
       const Type type)
      : output_{std::move(output)}, input_{std::move(input)}, type_{type} {}

  [[nodiscard]] auto &latch() const { return latch_; }

  [[nodiscard]] auto &input() const { return input_; }

  [[nodiscard]] auto &output() const { return output_; }

  [[nodiscard]] auto &intermediate() const { return intermediate_; }

  [[nodiscard]] auto &type() const { return type_; }

  virtual void operator()() const = 0;

  [[nodiscard]] virtual const char *name() const = 0;

 private:
  std::shared_ptr<std::latch> latch_{new std::latch{1}};

  std::vector<Tensor> output_;

  std::vector<ReadOnlyTensor> input_;

  mutable std::vector<at::Tensor> intermediate_;

  Type type_;
};
}  // namespace dllm
