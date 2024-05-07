#include "logger.h"
#include "optimizer/sgd.h"
#include "util.h"

namespace dllm::optimizer::Sgd {
namespace {
template <typename T>
__global__ void step(T* w, const T* dw, const T lr, std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  w[tid] -= lr * dw[tid];
}

template <typename Fn>
__inline__ __attribute__((always_inline)) void autoDispatch(Dtype dtype,
                                                            Fn&& fn) {
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
void stepKernel(cudaStream_t stream, Tensor1D& w, const Tensor1D& dw,
                double lr) {
  const auto size = cute::size(w.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min<decltype(size)>(128, size));
    dim3 grid(util::ceilDiv(size, std::min<decltype(size)>(128, size)));
    step<T><<<grid, block, 0, stream>>>(
        static_cast<T*>(w.data()), static_cast<const T*>(dw.data()), lr, size);
  };
  autoDispatch(w.dtype, f);
}
}  // namespace dllm::optimizer::Sgd
