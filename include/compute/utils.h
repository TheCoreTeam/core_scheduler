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
#include "tensor.h"
#include "threading/scheduler.h"

namespace cs::compute::Utils {
Tensor sum(const Scheduler &scheduler, const ReadOnlyTensor &input,
           IntArrayRef dim, bool keep_dim = false,
           c10::optional<at::ScalarType> dtype = c10::nullopt);

inline Tensor sum(const Scheduler &scheduler, const ReadOnlyTensor &input,
                  const int64_t dim, const bool keep_dim = false,
                  const c10::optional<at::ScalarType> dtype = c10::nullopt) {
  return sum(scheduler, input, IntArrayRef{dim}, keep_dim, dtype);
}

Tensor range(const Scheduler &scheduler, const at::Scalar &start,
             const at::Scalar &end, TensorOptions options = {});

Tensor arange(const Scheduler &scheduler, const at::Scalar &start,
              const at::Scalar &end, TensorOptions options = {});

Tensor arange(const Scheduler &scheduler, const at::Scalar &start,
              const at::Scalar &end, const at::Scalar &step,
              TensorOptions options = {});

Tensor randint(const Scheduler &scheduler, int64_t low, int64_t high,
               IntArrayRef size, TensorOptions options = at::kLong);

Tensor empty(const Scheduler &scheduler, IntArrayRef size,
             TensorOptions options = {});

Tensor empty_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

Tensor ones(const Scheduler &scheduler, IntArrayRef size,
            TensorOptions options = {});

Tensor ones_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

Tensor zeros(const Scheduler &scheduler, IntArrayRef size,
             TensorOptions options = {});

Tensor zeros_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

Tensor rand(const Scheduler &scheduler, IntArrayRef size,
            TensorOptions options = {});

Tensor rand_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

Tensor randn(const Scheduler &scheduler, IntArrayRef size,
             TensorOptions options = {});

Tensor randn_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

std::vector<Tensor> split(const Scheduler &scheduler, const ReadOnlyTensor &src,
                          const int64_t &split_size, const int64_t &dim);

Tensor view(const Scheduler &scheduler, const ReadOnlyTensor &input,
            IntArrayRef size);

Tensor broadcast_to(const Scheduler &scheduler, const ReadOnlyTensor &input,
                    IntArrayRef size);

Tensor cat(const Scheduler &scheduler, const std::vector<ReadOnlyTensor> &input,
           int64_t dim);

Tensor add(const Scheduler &scheduler, ReadOnlyTensor x, ReadOnlyTensor y);

void zero_(const Scheduler &scheduler, const Tensor &tensor);

Tensor clone(const Scheduler &scheduler, Tensor tensor);
}  // namespace cs::compute::Utils
