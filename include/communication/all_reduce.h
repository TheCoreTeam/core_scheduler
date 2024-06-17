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
#include "communication.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace cs::communication {
struct AllReduceBucket : Bucket {
  AllReduceBucket(int64_t byteThreshold, Operation operation);

  void push_back(const Scheduler& scheduler, const Comm& comm,
                 Tensor tensor) const;
};

struct AllReduce {
  static void runInplace(const Scheduler& scheduler, const Comm& comm,
                         const std::vector<Tensor>& tensors,
                         Operation operation);
};
}  // namespace cs::communication
