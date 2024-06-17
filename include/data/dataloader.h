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
#include <memory>

#include "data/dataset.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::data {
struct DataLoader {
  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

  [[nodiscard]] int64_t iterationsPerEpoch() const;

 protected:
  std::shared_ptr<Impl> impl_;
};

// This API is not stable
struct LlmDataLoader : DataLoader {
  std::unordered_map<std::string, Tensor> load(
      const Scheduler &scheduler) const;

  LlmDataLoader(const LlmDataset &dataset, int64_t batchSize,
                int64_t numWorkers, bool shuffle, int64_t rank,
                int64_t worldSize);
};
}  // namespace dllm::data
