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

#include <ATen/Dispatch.h>
#include <cuda_runtime.h>

#include "optimizer/adamw.h"
#include "tensor_impl.h"

namespace cs::optimizer {
namespace {
template <typename TA, typename TB>
constexpr __inline__ __attribute__((always_inline)) int ceil_div(TA a, TB b) {
  return (a + b - 1) / b;
}

template <typename UpT, typename T>
__device__ UpT cast_higher(const T v) {
  return static_cast<UpT>(v);
}

template <typename UpT, typename T>
__global__ void step(T *__restrict w, T *__restrict m, T *__restrict v,
                     const T *__restrict dw, const UpT lr, const UpT beta1,
                     const UpT beta2, const UpT inv_one_minus_beta1_pow_t,
                     const UpT inv_one_minus_beta2_pow_t, const UpT eps,
                     const UpT weigt_decay, const std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const auto one = cast_higher<UpT>(1);
  const auto g_t = cast_higher<UpT>(dw[tid]);
  auto theta_t = cast_higher<UpT>(w[tid]) - cast_higher<UpT>(lr) *
                                                cast_higher<UpT>(weigt_decay) *
                                                cast_higher<UpT>(w[tid]);
  const auto m_t = cast_higher<UpT>(beta1) * cast_higher<UpT>(m[tid]) +
                   (one - cast_higher<UpT>(beta1)) * g_t;
  m[tid] = m_t;
  const auto v_t = cast_higher<UpT>(beta2) * cast_higher<UpT>(v[tid]) +
                   (one - cast_higher<UpT>(beta2)) * g_t * g_t;
  v[tid] = v_t;
  const auto m_hat_t = m_t * cast_higher<UpT>(inv_one_minus_beta1_pow_t);
  const auto v_hat_t = v_t * cast_higher<UpT>(inv_one_minus_beta2_pow_t);
  w[tid] = theta_t - cast_higher<UpT>(lr) * m_hat_t /
                         (std::sqrt(v_hat_t) + cast_higher<UpT>(eps));
}

template <typename UpT, typename T>
__global__ void step(T *__restrict w, T *__restrict m, T *__restrict v,
                     T *__restrict vMax, const T *__restrict dw, const UpT lr,
                     const UpT beta1, const UpT beta2,
                     const UpT inv_one_minus_beta1_pow_t,
                     const UpT inv_one_minus_beta2_pow_t, const UpT eps,
                     const UpT weigt_decay, const std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const auto one = cast_higher<UpT>(static_cast<T>(1));
  const auto g_t = cast_higher<UpT>(dw[tid]);
  auto theta_t = cast_higher<UpT>(w[tid]) - cast_higher<UpT>(lr) *
                                                cast_higher<UpT>(weigt_decay) *
                                                cast_higher<UpT>(w[tid]);
  const auto m_t = cast_higher<UpT>(beta1) * cast_higher<UpT>(m[tid]) +
                   (one - cast_higher<UpT>(beta1)) * g_t;
  m[tid] = m_t;
  const auto v_t = cast_higher<UpT>(beta2) * cast_higher<UpT>(v[tid]) +
                   (one - cast_higher<UpT>(beta2)) * g_t * g_t;
  v[tid] = v_t;
  const auto m_hat_t = m_t * cast_higher<UpT>(inv_one_minus_beta1_pow_t);
  const auto v_hat_t = v_t * cast_higher<UpT>(inv_one_minus_beta2_pow_t);
  const auto v_hat_max_t = std::max(cast_higher<UpT>(vMax[tid]), v_hat_t);
  vMax[tid] = v_hat_max_t;
  w[tid] = theta_t - cast_higher<UpT>(lr) * m_hat_t /
                         (std::sqrt(v_hat_max_t) + cast_higher<UpT>(eps));
}
}  // namespace

void stepKernel(cudaStream_t stream, const AdamW::State::Options &options,
                const Tensor &w, const Tensor &m, const Tensor &v,
                const ReadOnlyTensor &dw) {
  const auto size = [&] {
    const auto sizes = dw.impl()->tensor().sizes();
    int64_t s = 1;
    for (const auto e : sizes) {
      s *= e;
    }
    return s;
  }();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      dw.impl()->tensor().scalar_type(), "AdamW w/o Amsgrad", [&] {
        using T = scalar_t;
        using UpT =
            std::conditional_t<std::is_same_v<T, double>, double, float>;
        dim3 block(std::min<decltype(size)>(128, size));
        dim3 grid(ceil_div(size, std::min<decltype(size)>(128, size)));
        step<UpT, T><<<grid, block, 0, stream>>>(
            w.impl()->tensor().data_ptr<T>(), m.impl()->tensor().data_ptr<T>(),
            v.impl()->tensor().data_ptr<T>(), dw.impl()->tensor().data_ptr<T>(),
            options.lr, options.beta1, options.beta2,
            1. / (1. - std::pow(options.beta1, options.t)),
            1. / (1. - std::pow(options.beta2, options.t)), options.eps,
            options.weight_decay, size);
      });
}

void stepKernelAmsgrad(cudaStream_t stream,
                       const AdamW::State::Options &options, const Tensor &w,
                       const Tensor &m, const Tensor &v, const Tensor &vMax,
                       const ReadOnlyTensor &dw) {
  const auto size = [&] {
    const auto sizes = dw.impl()->tensor().sizes();
    int64_t s = 1;
    for (const auto e : sizes) {
      s *= e;
    }
    return s;
  }();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      dw.impl()->tensor().scalar_type(), "AdamW w/ Amsgrad", [&] {
        using T = scalar_t;
        using UpT =
            std::conditional_t<std::is_same_v<T, double>, double, float>;
        dim3 block(std::min<decltype(size)>(128, size));
        dim3 grid(ceil_div(size, std::min<decltype(size)>(128, size)));
        step<UpT, T><<<grid, block, 0, stream>>>(
            w.impl()->tensor().data_ptr<T>(), m.impl()->tensor().data_ptr<T>(),
            v.impl()->tensor().data_ptr<T>(),
            vMax.impl()->tensor().data_ptr<T>(),
            dw.impl()->tensor().data_ptr<T>(), options.lr, options.beta1,
            options.beta2, 1. / (1. - std::pow(options.beta1, options.t)),
            1. / (1. - std::pow(options.beta2, options.t)), options.eps,
            options.weight_decay, size);
      });
}
}  // namespace cs::optimizer
