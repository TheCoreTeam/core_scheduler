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
#include "threading/scheduler.h"
#include "threading/task.h"

namespace cs {
struct Scheduler::Impl {
  Impl(int64_t deviceRank);

  virtual ~Impl() = default;

  virtual void submit(Task &&task);

  [[nodiscard]] int64_t device_rank() const;

 protected:
  const int64_t device_rank_;
};
}  // namespace cs