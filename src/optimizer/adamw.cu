#include <ATen/Dispatch.h>

#include "internal_utils.h"
#include "optimizer/adamw.h"
#include "tensor_friend.h"

namespace dllm::optimizer {
namespace {
template <typename T>
__device__ float cast_higher(const T v) {
  return static_cast<float>(v);
}

__device__ double cast_higher(
    const c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Double> v) {
  return v;
}

template <typename T>
__global__ void step(T *__restrict w, T *__restrict m, T *__restrict v,
                     const T *__restrict dw, const T lr, const T beta1,
                     const T beta2, const T inv_one_minus_beta1_pow_t,
                     const T inv_one_minus_beta2_pow_t, const T eps,
                     const T weigt_decay, const std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const auto one = cast_higher(static_cast<T>(1));
  const auto g_t = cast_higher(dw[tid]);
  auto theta_t = cast_higher(w[tid]) - cast_higher(lr) *
                                           cast_higher(weigt_decay) *
                                           cast_higher(w[tid]);
  const auto m_t = cast_higher(beta1) * cast_higher(m[tid]) +
                   (one - cast_higher(beta1)) * g_t;
  m[tid] = m_t;
  const auto v_t = cast_higher(beta2) * cast_higher(v[tid]) +
                   (one - cast_higher(beta2)) * g_t * g_t;
  v[tid] = v_t;
  const auto m_hat_t = m_t * cast_higher(inv_one_minus_beta1_pow_t);
  const auto v_hat_t = v_t * cast_higher(inv_one_minus_beta2_pow_t);
  w[tid] = theta_t -
           cast_higher(lr) * m_hat_t / (std::sqrt(v_hat_t) + cast_higher(eps));
}

template <typename T>
__global__ void step(T *__restrict w, T *__restrict m, T *__restrict v,
                     T *__restrict vMax, const T *__restrict dw, const T lr,
                     const T beta1, const T beta2,
                     const T inv_one_minus_beta1_pow_t,
                     const T inv_one_minus_beta2_pow_t, const T eps,
                     const T weigt_decay, const std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const auto one = cast_higher(static_cast<T>(1));
  const auto g_t = cast_higher(dw[tid]);
  auto theta_t = cast_higher(w[tid]) - cast_higher(lr) *
                                           cast_higher(weigt_decay) *
                                           cast_higher(w[tid]);
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
  w[tid] = theta_t - cast_higher(lr) * m_hat_t /
                         (std::sqrt(v_hat_max_t) + cast_higher(eps));
}
}  // namespace

void stepKernel(cudaStream_t stream, const AdamW::State::Options &options,
                const std::shared_ptr<Tensor> &w,
                const std::shared_ptr<Tensor> &m,
                const std::shared_ptr<Tensor> &v,
                const std::shared_ptr<const ReadOnlyTensor> &dw) {
  const auto size = [&] {
    const auto sizes = DLLM_EXTRACT_TENSOR(dw).sizes();
    int64_t s = 1;
    for (const auto e : sizes) {
      s *= e;
    }
    return s;
  }();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      DLLM_EXTRACT_TENSOR(dw).scalar_type(), "AdamW w/o Amsgrad", [&] {
        using T = scalar_t;
        dim3 block(std::min<decltype(size)>(128, size));
        dim3 grid(util::ceil_div(size, std::min<decltype(size)>(128, size)));
        step<T><<<grid, block, 0, stream>>>(
            DLLM_EXTRACT_TENSOR(w).data_ptr<T>(),
            DLLM_EXTRACT_TENSOR(m).data_ptr<T>(),
            DLLM_EXTRACT_TENSOR(v).data_ptr<T>(),
            DLLM_EXTRACT_TENSOR(dw).data_ptr<T>(), options.lr, options.beta1,
            options.beta2, 1. / (1. - std::pow(options.beta1, options.t)),
            1. / (1. - std::pow(options.beta2, options.t)), options.eps,
            options.weight_decay, size);
      });
}

void stepKernelAmsgrad(cudaStream_t stream,
                       const AdamW::State::Options &options,
                       const std::shared_ptr<Tensor> &w,
                       const std::shared_ptr<Tensor> &m,
                       const std::shared_ptr<Tensor> &v,
                       const std::shared_ptr<Tensor> &vMax,
                       const std::shared_ptr<const ReadOnlyTensor> &dw) {
  const auto size = [&] {
    const auto sizes = DLLM_EXTRACT_TENSOR(dw).sizes();
    int64_t s = 1;
    for (const auto e : sizes) {
      s *= e;
    }
    return s;
  }();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      DLLM_EXTRACT_TENSOR(dw).scalar_type(), "AdamW w/o Amsgrad", [&] {
        using T = scalar_t;
        dim3 block(std::min<decltype(size)>(128, size));
        dim3 grid(util::ceil_div(size, std::min<decltype(size)>(128, size)));
        step<T><<<grid, block, 0, stream>>>(
            DLLM_EXTRACT_TENSOR(w).data_ptr<T>(),
            DLLM_EXTRACT_TENSOR(m).data_ptr<T>(),
            DLLM_EXTRACT_TENSOR(v).data_ptr<T>(),
            DLLM_EXTRACT_TENSOR(vMax).data_ptr<T>(),
            DLLM_EXTRACT_TENSOR(dw).data_ptr<T>(), options.lr, options.beta1,
            options.beta2, 1. / (1. - std::pow(options.beta1, options.t)),
            1. / (1. - std::pow(options.beta2, options.t)), options.eps,
            options.weight_decay, size);
      });
}
}  // namespace dllm::optimizer
