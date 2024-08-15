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
#ifdef CORE_SCHEDULER_ENABLE_INTERNAL_BUILD
// This is an internal file, never use it unless you know what you are doing
#include <ATen/core/TensorBody.h>
#include <cuda_runtime.h>

#include "tensor.h"
#include "threading/event.h"
#include "threading/task_impl.h"

namespace cs {
struct Event;

struct ReadOnlyTensor::Impl {
  Task::Impl::Priority priority_;
  auto &priority() { return priority_; }
  auto &priority() const { return priority_; }

  Event event_;
  auto &event() { return event_; }
  auto &event() const { return event_; }

  struct {
    bool enable = false;
    c10::ScalarType dtype;
  } auto_cast_;
  auto &auto_cast() { return auto_cast_; }
  auto &auto_cast() const { return auto_cast_; }

  Impl() = default;

  [[nodiscard]] auto options() const { return tensor_.options(); }

  [[nodiscard]] auto sizes() const { return tensor_.sizes(); }

  [[nodiscard]] auto size(const int64_t dim) const { return tensor_.size(dim); }

  [[nodiscard]] auto numel() const { return tensor_.numel(); }

  auto &tensor() { return tensor_; }

  at::Tensor tensor_{};
};
}  // namespace cs
#else
#error "You should not include this file in your program!"
#endif
