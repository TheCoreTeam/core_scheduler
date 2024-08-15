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

#include "threading/task.h"

#include "threading/task_impl.h"

namespace cs {
Task::Task(std::shared_ptr<Impl> impl) : impl_{std::move(impl)} {}

void Task::operator()() const { impl_->operator()(); }

const std::shared_ptr<Task::Impl>& Task::impl() const { return impl_; }

void Task::reset() { impl_.reset(); }
}  // namespace cs
