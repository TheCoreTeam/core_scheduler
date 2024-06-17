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

#include <arpa/inet.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "logger.h"
#include "threading/scheduler_impl.h"

namespace dllm {
const std::shared_ptr<Scheduler::Impl>& Scheduler::impl() const {
  return impl_;
}
int64_t Scheduler::deviceRank() const { return impl()->deviceRank(); }

Scheduler::Impl::Impl(const int64_t deviceRank) : deviceRank_{deviceRank} {}

void Scheduler::Impl::submit(Task&& task) {
  DLLM_ASSERT_TRUE(false, "Wrong task - Scheduler pair");
}

int64_t Scheduler::Impl::deviceRank() const { return deviceRank_; }
}  // namespace dllm
