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

#include "export.h"

namespace cs::data {
struct Dataset {
  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

 protected:
  std::shared_ptr<Impl> impl_;
};

// This API is not stable
struct CS_API LlmDataset : Dataset {
  explicit LlmDataset(const std::string &directory);

  [[nodiscard]] std::int64_t size() const;

  [[nodiscard]] const std::vector<std::string> &files() const;
};
}  // namespace cs::data
