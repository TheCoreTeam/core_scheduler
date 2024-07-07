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
CS_API Tensor sum(const Scheduler &scheduler, const ReadOnlyTensor &input,
                  IntArrayRef dim, bool keep_dim = false,
                  c10::optional<at::ScalarType> dtype = c10::nullopt);

inline Tensor sum(const Scheduler &scheduler, const ReadOnlyTensor &input,
                  const int64_t dim, const bool keep_dim = false,
                  const c10::optional<at::ScalarType> dtype = c10::nullopt) {
  return sum(scheduler, input, IntArrayRef{dim}, keep_dim, dtype);
}

CS_API Tensor range(const Scheduler &scheduler, const at::Scalar &start,
                    const at::Scalar &end, TensorOptions options = {});

CS_API Tensor arange(const Scheduler &scheduler, const at::Scalar &start,
                     const at::Scalar &end, TensorOptions options = {});

CS_API Tensor arange(const Scheduler &scheduler, const at::Scalar &start,
                     const at::Scalar &end, const at::Scalar &step,
                     TensorOptions options = {});

CS_API Tensor randint(const Scheduler &scheduler, int64_t low, int64_t high,
                      IntArrayRef size, TensorOptions options = at::kLong);

CS_API Tensor empty(const Scheduler &scheduler, IntArrayRef size,
                    TensorOptions options = {});

CS_API Tensor empty_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

CS_API Tensor ones(const Scheduler &scheduler, IntArrayRef size,
                   TensorOptions options = {});

CS_API Tensor ones_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

CS_API Tensor zeros(const Scheduler &scheduler, IntArrayRef size,
                    TensorOptions options = {});

CS_API Tensor zeros_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

CS_API Tensor rand(const Scheduler &scheduler, IntArrayRef size,
                   TensorOptions options = {});

CS_API Tensor rand_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

CS_API Tensor randn(const Scheduler &scheduler, IntArrayRef size,
                    TensorOptions options = {});

CS_API Tensor randn_like(const Scheduler &scheduler, const ReadOnlyTensor &src);

CS_API std::vector<Tensor> split(const Scheduler &scheduler,
                                 const ReadOnlyTensor &src,
                                 const int64_t &split_size, const int64_t &dim);

CS_API Tensor view(const Scheduler &scheduler, const ReadOnlyTensor &input,
                   IntArrayRef size);

CS_API Tensor as_strided(const Scheduler &scheduler,
                         const ReadOnlyTensor &input, IntArrayRef size,
                         IntArrayRef stride,
                         optional<int64_t> storage_offset = c10::nullopt);

CS_API Tensor broadcast_to(const Scheduler &scheduler,
                           const ReadOnlyTensor &input, IntArrayRef size);

CS_API Tensor cat(const Scheduler &scheduler,
                  const std::vector<ReadOnlyTensor> &input, int64_t dim);

CS_API Tensor add(const Scheduler &scheduler, ReadOnlyTensor x,
                  ReadOnlyTensor y);

CS_API void zero_(const Scheduler &scheduler, const Tensor &tensor);

CS_API void uniform_(const Scheduler &scheduler, const Tensor &tensor,
                     double from = 0, double to = 1);

CS_API void normal_(const Scheduler &scheduler, const Tensor &tensor,
                    double mean = 0, double std = 1);

CS_API Tensor clone(const Scheduler &scheduler, const Tensor &tensor);

CS_API Tensor linalg_vector_norm(
    const Scheduler &scheduler, const ReadOnlyTensor &self,
    const Scalar &ord = 2, at::OptionalIntArrayRef dim = c10::nullopt,
    bool keepdim = false, c10::optional<at::ScalarType> dtype = c10::nullopt);
}  // namespace cs::compute::Utils
