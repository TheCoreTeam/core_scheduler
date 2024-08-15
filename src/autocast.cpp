/*
 * Copyright (c) 2024 The Core Team
 *
 * Licensed under the Apache License, Version 2.0;
 * You may not use this file except in compliance with the License.
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

#include "autocast.h"

#include <ATen/autocast_mode.h>

#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::autocast {
ContextGuard::ContextGuard(const Scheduler& scheduler,
                           const c10::ScalarType& dtype)
    : scheduler_{scheduler} {
  at::autocast::clear_cache();
  at::autocast::set_enabled(true);
  at::autocast::set_autocast_cache_enabled(true);
  at::autocast::set_autocast_gpu_dtype(dtype);

  struct Impl : Task::Impl {
    const c10::ScalarType dtype;

    explicit Impl(const c10::ScalarType& dtype)
        : Task::Impl{{}, {}, kMain, kConfig}, dtype{dtype} {}
    void operator()() const override {
      at::autocast::clear_cache();
      at::autocast::set_enabled(true);
      at::autocast::set_autocast_cache_enabled(true);
      at::autocast::set_autocast_gpu_dtype(dtype);
    }
    [[nodiscard]] const char* name() const override {
      return "cs::autocast::enter";
    }
  };

  scheduler.impl()->submit(Task{std::make_shared<Impl>(dtype)});
}

ContextGuard::~ContextGuard() {
  at::autocast::set_enabled(false);

  struct Impl : Task::Impl {
    explicit Impl() : Task::Impl{{}, {}, kMain, kConfig} {}
    void operator()() const override { at::autocast::set_enabled(false); }
    [[nodiscard]] const char* name() const override {
      return "cs::autocast::exit";
    }
  };

  scheduler_.impl()->submit(Task{std::make_shared<Impl>()});
}
}  // namespace cs::autocast
