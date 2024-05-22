#include "optimizer/adamw.h"
#include "util.h"

namespace std {
__device__ nv_half sqrt(const nv_half &x) { return hsqrt(x); }
__device__ nv_bfloat16 sqrt(const nv_bfloat16 &x) { return hsqrt(x); }
}  // namespace std

namespace dllm::optimizer {
namespace {
template <typename T>
__global__ void step(T *__restrict w, T *__restrict m, T *__restrict v,
                     const T *__restrict dw, const T lr, const T beta1,
                     const T beta2, const T inv_one_minus_beta1_pow_t,
                     const T inv_one_minus_beta2_pow_t, const T eps,
                     const T weigt_decay, std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const auto g_t = dw[tid];
  auto theta_t = w[tid] - lr * weigt_decay * w[tid];
  const auto m_t = beta1 * m[tid] + (static_cast<T>(1) - beta1) * g_t;
  m[tid] = m_t;
  const auto v_t = beta2 * v[tid] + (static_cast<T>(1) - beta2) * g_t * g_t;
  v[tid] = v_t;
  const auto m_hat_t = m_t * inv_one_minus_beta1_pow_t;
  const auto v_hat_t = v_t * inv_one_minus_beta2_pow_t;
  w[tid] = theta_t - lr * m_hat_t / (std::sqrt(v_hat_t) + eps);
}

template <typename T>
__global__ void step(T *__restrict w, T *__restrict m, T *__restrict v,
                     T *__restrict vMax, const T *__restrict dw, const T lr,
                     const T beta1, const T beta2,
                     const T inv_one_minus_beta1_pow_t,
                     const T inv_one_minus_beta2_pow_t, const T eps,
                     const T weigt_decay, std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const auto g_t = dw[tid];
  auto theta_t = w[tid] - lr * weigt_decay * w[tid];
  const auto m_t = beta1 * m[tid] + (static_cast<T>(1) - beta1) * g_t;
  m[tid] = m_t;
  const auto v_t = beta2 * v[tid] + (static_cast<T>(1) - beta2) * g_t * g_t;
  v[tid] = v_t;
  const auto m_hat_t = m_t * inv_one_minus_beta1_pow_t;
  const auto v_hat_t = v_t * inv_one_minus_beta2_pow_t;
  const auto v_hat_max_t = std::max(vMax[tid], v_hat_t);
  vMax[tid] = v_hat_max_t;
  w[tid] = theta_t - lr * m_hat_t / (std::sqrt(v_hat_max_t) + eps);
}

template <typename Fn>
__inline__ __attribute__((always_inline)) void autoDispatch(Dtype dtype,
                                                            Fn &&fn) {
  switch (dtype) {
    case R_64F:
      fn(double{0});
      return;
    case R_32F:
      fn(float{0});
      return;
    case R_16F:
      fn(nv_half{0});
      return;
    case R_16BF:
      fn(nv_bfloat16{0});
      return;
    default:
      return;
  }
}
}  // namespace

template <bool amsgrad>
void stepKernel(cudaStream_t stream,
                const typename AdamW<amsgrad>::State &state, Tensor1D &w,
                const Tensor1D &dw);

template <>
void stepKernel<false>(cudaStream_t stream, const AdamW<false>::State &state,
                       Tensor1D &w, const Tensor1D &dw) {
  const auto size = cute::size(w.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min<decltype(size)>(128, size));
    dim3 grid(util::ceilDiv(size, std::min<decltype(size)>(128, size)));
    step<T><<<grid, block, 0, stream>>>(
        static_cast<T *>(w.data()), static_cast<T *>(state.m->data()),
        static_cast<T *>(state.v->data()), static_cast<const T *>(dw.data()),
        state.lr, state.beta1, state.beta2,
        1. / (1. - std::pow(state.beta1, state.t)),
        1. / (1. - std::pow(state.beta2, state.t)), state.eps,
        state.weight_decay, size);
  };
  autoDispatch(w.dtype, f);
}

template <>
void stepKernel<true>(cudaStream_t stream, const AdamW<true>::State &state,
                      Tensor1D &w, const Tensor1D &dw) {
  const auto size = cute::size(w.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min<decltype(size)>(128, size));
    dim3 grid(util::ceilDiv(size, std::min<decltype(size)>(128, size)));
    step<T><<<grid, block, 0, stream>>>(
        static_cast<T *>(w.data()), static_cast<T *>(state.m->data()),
        static_cast<T *>(state.v->data()), static_cast<T *>(state.vMax->data()),
        static_cast<const T *>(dw.data()), state.lr, state.beta1, state.beta2,
        1. / (1. - std::pow(state.beta1, state.t)),
        1. / (1. - std::pow(state.beta2, state.t)), state.eps,
        state.weight_decay, size);
  };
  autoDispatch(w.dtype, f);
}
}  // namespace dllm::optimizer
