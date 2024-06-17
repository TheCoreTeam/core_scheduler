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

namespace dllm {
struct Event {
  struct Impl;

  Event();

  explicit Event(std::shared_ptr<Impl> impl);

  ~Event();

  void block() const;

  void record() const;

  [[nodiscard]] bool query() const;

  void synchronize() const;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

 private:
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm