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

#include <c10/cuda/CUDAStream.h>

#include "logger.h"
#include "threading/event_impl.h"

namespace cs {
namespace {
auto &getEventMutex() {
  static std::mutex mutex;
  return mutex;
}

auto &getEventImpls() {
  static std::vector<std::shared_ptr<Event::Impl>> events;
  return events;
}

auto popEvent() {
  std::lock_guard lock{getEventMutex()};
  if (auto &events = getEventImpls(); events.empty()) {
    return std::make_shared<Event::Impl>();
  } else {
    auto event = std::move(events.back());
    events.pop_back();
    return event;
  }
}

auto pushEvent(std::shared_ptr<Event::Impl> event) {
  std::lock_guard lock{getEventMutex()};
  getEventImpls().push_back(std::move(event));
}
}  // namespace

void Event::Impl::block() { event.block(c10::cuda::getCurrentCUDAStream()); }

void Event::Impl::record() { event.record(); }

bool Event::Impl::query() const { return event.query(); }

void Event::Impl::synchronize() const { event.synchronize(); }

Event::Event() : impl_{popEvent()} {}

Event::Event(std::shared_ptr<Impl> impl) : impl_{std::move(impl)} {}

Event::~Event() {
  if (impl_ != nullptr && impl_.use_count() == 1) {
    pushEvent(std::move(impl_));
  }
}

void Event::block() const { impl_->block(); }

void Event::record() const { impl_->record(); }

bool Event::query() const { return impl_->query(); }

void Event::synchronize() const { impl_->synchronize(); }

const std::shared_ptr<Event::Impl> &Event::impl() const { return impl_; }
}  // namespace cs
