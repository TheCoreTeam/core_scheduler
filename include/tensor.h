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
#include <ATen/core/TensorBody.h>

#include <memory>

namespace cs {
using IntArrayRef = at::IntArrayRef;

using IntArray = c10::SmallVector<IntArrayRef::value_type>;

using TensorOptions = at::TensorOptions;

template <typename ELement>
using optional = c10::optional<ELement>;

struct ReadOnlyTensor {
  ReadOnlyTensor();

  [[nodiscard]] TensorOptions options() const;

  [[nodiscard]] IntArrayRef sizes() const;

  [[nodiscard]] int64_t size(int64_t dim) const;

  [[nodiscard]] int64_t numel() const;

  void wait() const;

  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

  void reset();

 protected:
  std::shared_ptr<Impl> impl_;
};

struct Tensor : ReadOnlyTensor {
  Tensor() = default;

  void wait() const;
};

TORCH_API std::ostream &print(std::ostream &stream,
                              const ReadOnlyTensor &tensor, int64_t linesize);

static std::ostream &operator<<(std::ostream &out, const ReadOnlyTensor &t) {
  return print(out, t, 80);
}
}  // namespace cs

namespace at {
bool allclose(const cs::ReadOnlyTensor &t1, const at::Tensor &t2,
              double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
bool allclose(const cs::ReadOnlyTensor &t1, const cs::ReadOnlyTensor &t2,
              double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
bool allclose(const at::Tensor &t1, const cs::ReadOnlyTensor &t2,
              double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
}  // namespace at
