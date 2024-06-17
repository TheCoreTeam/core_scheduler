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
#include <ATen/cuda/CUDAEvent.h>

#include "threading/event.h"

namespace cs {
struct Event::Impl {
  void block();

  void record();

  [[nodiscard]] bool query() const;

  void synchronize() const;

 private:
  at::cuda::CUDAEvent event;
};
}  // namespace cs
