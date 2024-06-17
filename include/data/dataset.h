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
#include <cstdint>
#include <memory>
#include <vector>

namespace dllm::data {
struct Dataset {
  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

 protected:
  std::shared_ptr<Impl> impl_;
};

// This API is not stable
struct LlmDataset : Dataset {
  explicit LlmDataset(const std::string &directory);

  void fillBatch(const std::vector<std::int64_t *> &ptrs,
                 const std::vector<std::int64_t> &ld, std::int64_t startingRow,
                 std::int64_t batchSize) const;

  [[nodiscard]] std::int64_t attributeNum() const;

  [[nodiscard]] const std::vector<std::string> &attributeNames() const;

  [[nodiscard]] std::int64_t rows() const;

  [[nodiscard]] std::int64_t cols() const;
};
}  // namespace dllm::data
