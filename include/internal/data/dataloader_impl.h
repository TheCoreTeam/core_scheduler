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
#include "data/dataloader.h"

namespace cs::data {
struct DataLoader::Impl {
  Impl(int64_t batchSize, int64_t numWorkers, bool shuffle, int64_t rank,
       int64_t worldSize);

  const int64_t batchSize;
  const int64_t num_workers;
  const bool shuffle;
  const int64_t rank;
  const int64_t world_size;

  [[nodiscard]] virtual int64_t iterations_per_epoch() const = 0;

  virtual ~Impl() = default;
};
}  // namespace cs::data
