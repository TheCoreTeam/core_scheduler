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

#include "tensor.h"

#include <ATen/ops/allclose.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>

#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/task_impl.h"

namespace cs {
template <typename T1, typename T2>
static bool allclose_impl(const T1 &t1_, const T2 &t2_, const double rtol,
                          const double atol, const bool equal_nan) {
  CS_NVTX_RANGE_FN("cs::allclose");
  at::Tensor t1, t2;
  torch::cuda::synchronize();
  if constexpr (std::is_same_v<T1, at::Tensor>) {
    t1 = t1_;
  } else {
    t1_.wait();
    t1 = t1_.impl()->tensor();
  }
  if constexpr (std::is_same_v<T2, at::Tensor>) {
    t2 = t2_;
  } else {
    t2_.wait();
    t2 = t2_.impl()->tensor();
  }
  return at::allclose(t1, t2, rtol, atol, equal_nan);
}

void ReadOnlyTensor::wait() const { impl()->event().synchronize(); }

const std::shared_ptr<ReadOnlyTensor::Impl> &ReadOnlyTensor::impl() const {
  return impl_;
}
void ReadOnlyTensor::reset() { *this = ReadOnlyTensor{}; }

void Tensor::wait() const { static_cast<const ReadOnlyTensor *>(this)->wait(); }

std::ostream &print(std::ostream &stream, const ReadOnlyTensor &tensor,
                    const int64_t linesize) {
  tensor.wait();
  return at::print(stream, tensor.impl()->tensor(), linesize);
}

int64_t ReadOnlyTensor::numel() const { return impl_->numel(); }

int64_t ReadOnlyTensor::size(const int64_t dim) const {
  return impl_->size(dim);
}

IntArrayRef ReadOnlyTensor::sizes() const { return impl_->sizes(); }

ReadOnlyTensor::ReadOnlyTensor() : impl_{std::make_shared<Impl>()} {}

TensorOptions ReadOnlyTensor::options() const { return impl_->options(); }

}  // namespace cs

namespace at {
bool allclose(const cs::ReadOnlyTensor &t1, const at::Tensor &t2,
              const double rtol, const double atol, const bool equal_nan) {
  return cs::allclose_impl(t1, t2, rtol, atol, equal_nan);
}
bool allclose(const cs::ReadOnlyTensor &t1, const cs::ReadOnlyTensor &t2,
              const double rtol, const double atol, const bool equal_nan) {
  return cs::allclose_impl(t1, t2, rtol, atol, equal_nan);
}
bool allclose(const at::Tensor &t1, const cs::ReadOnlyTensor &t2,
              const double rtol, const double atol, const bool equal_nan) {
  return cs::allclose_impl(t1, t2, rtol, atol, equal_nan);
}
}  // namespace at
