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

#include "optimizer/amp_adamw.h"
#include "tensor_impl.h"

namespace cs::optimizer {
namespace {
template <typename TA, typename TB>
constexpr __inline__ __attribute__((always_inline)) int ceil_div(TA a, TB b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ float cast_higher(const T v) {
  return static_cast<float>(v);
}

template <typename T>
__global__ void step(T *__restrict w, float *__restrict wFp32,
                     float *__restrict m, float *__restrict v,
                     const T *__restrict dw, const float lr, const float beta1,
                     const float beta2, const float inv_one_minus_beta1_pow_t,
                     const float inv_one_minus_beta2_pow_t, const float eps,
                     const float weigt_decay, const std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const auto one = cast_higher(1);
  const auto g_t = cast_higher(dw[tid]);
  auto theta_t = cast_higher(wFp32[tid]) - cast_higher(lr) *
                                               cast_higher(weigt_decay) *
                                               cast_higher(wFp32[tid]);
  const auto m_t = cast_higher(beta1) * cast_higher(m[tid]) +
                   (one - cast_higher(beta1)) * g_t;
  m[tid] = m_t;
  const auto v_t = cast_higher(beta2) * cast_higher(v[tid]) +
                   (one - cast_higher(beta2)) * g_t * g_t;
  v[tid] = v_t;
  const auto m_hat_t = m_t * cast_higher(inv_one_minus_beta1_pow_t);
  const auto v_hat_t = v_t * cast_higher(inv_one_minus_beta2_pow_t);
  wFp32[tid] = theta_t - cast_higher(lr) * m_hat_t /
                             (std::sqrt(v_hat_t) + cast_higher(eps));
  w[tid] = wFp32[tid];
}

template <typename T>
__global__ void step(T *__restrict w, float *__restrict wFp32,
                     float *__restrict m, float *__restrict v,
                     float *__restrict vMax, const T *__restrict dw,
                     const float lr, const float beta1, const float beta2,
                     const float inv_one_minus_beta1_pow_t,
                     const float inv_one_minus_beta2_pow_t, const float eps,
                     const float weigt_decay, const std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const auto one = cast_higher(static_cast<T>(1));
  const auto g_t = cast_higher(dw[tid]);
  auto theta_t = cast_higher(wFp32[tid]) - cast_higher(lr) *
                                               cast_higher(weigt_decay) *
                                               cast_higher(wFp32[tid]);
  const auto m_t = cast_higher(beta1) * cast_higher(m[tid]) +
                   (one - cast_higher(beta1)) * g_t;
  m[tid] = m_t;
  const auto v_t = cast_higher(beta2) * cast_higher(v[tid]) +
                   (one - cast_higher(beta2)) * g_t * g_t;
  v[tid] = v_t;
  const auto m_hat_t = m_t * cast_higher(inv_one_minus_beta1_pow_t);
  const auto v_hat_t = v_t * cast_higher(inv_one_minus_beta2_pow_t);
  const auto v_hat_max_t = std::max(cast_higher(vMax[tid]), v_hat_t);
  vMax[tid] = v_hat_max_t;
  wFp32[tid] = theta_t - cast_higher(lr) * m_hat_t /
                             (std::sqrt(v_hat_max_t) + cast_higher(eps));
  w[tid] = wFp32[tid];
}
}  // namespace

void ampStepKernel(cudaStream_t stream, const AmpAdamW::State::Options &options,
                   const Tensor &w, const Tensor &wFp32, const Tensor &m,
                   const Tensor &v, const ReadOnlyTensor &dw) {
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
      dw.impl()->tensor().scalar_type(), "AmpAdamW w/o Amsgrad", [&] {
        using T = scalar_t;
        dim3 block(std::min<decltype(size)>(128, size));
        dim3 grid(ceil_div(size, std::min<decltype(size)>(128, size)));
        step<T><<<grid, block, 0, stream>>>(
            w.impl()->tensor().data_ptr<T>(),
            wFp32.impl()->tensor().data_ptr<float>(),
            m.impl()->tensor().data_ptr<float>(),
            v.impl()->tensor().data_ptr<float>(),
            dw.impl()->tensor().data_ptr<T>(), options.lr, options.beta1,
            options.beta2, 1. / (1. - std::pow(options.beta1, options.t)),
            1. / (1. - std::pow(options.beta2, options.t)), options.eps,
            options.weight_decay, size);
      });
}

void ampStepKernelAmsgrad(cudaStream_t stream,
                          const AmpAdamW::State::Options &options,
                          const Tensor &w, const Tensor &m, const Tensor &wFp32,
                          const Tensor &v, const Tensor &vMax,
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
      dw.impl()->tensor().scalar_type(), "AmpAdamW w/ Amsgrad", [&] {
        using T = scalar_t;
        dim3 block(std::min<decltype(size)>(128, size));
        dim3 grid(ceil_div(size, std::min<decltype(size)>(128, size)));
        step<T><<<grid, block, 0, stream>>>(
            w.impl()->tensor().data_ptr<T>(),
            wFp32.impl()->tensor().data_ptr<float>(),
            m.impl()->tensor().data_ptr<float>(),
            v.impl()->tensor().data_ptr<float>(),
            vMax.impl()->tensor().data_ptr<float>(),
            dw.impl()->tensor().data_ptr<T>(), options.lr, options.beta1,
            options.beta2, 1. / (1. - std::pow(options.beta1, options.t)),
            1. / (1. - std::pow(options.beta2, options.t)), options.eps,
            options.weight_decay, size);
      });
}
}  // namespace cs::optimizer
