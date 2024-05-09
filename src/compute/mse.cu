#include <thrust/transform.h>

#include "compute/mse.h"
#include "util.h"

namespace dllm::compute::Mse {
namespace {
template <typename T>
struct squareDiff {
  __host__ __device__ float operator()(const T& x, const T& y) const {
    T diff = x - y;
    return diff * diff;
  }
};

template <typename T>
__global__ void grad(T* dx, const T* x, const T* y, std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  T grad = static_cast<T>(2) * (x[tid] - y[tid]);
  dx[tid] = grad;
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

void forwardKernel(cudaStream_t stream, Tensor1D& error, const Tensor1D& x,
                   const Tensor1D& y) {
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    thrust::transform(thrust::cuda::par.on(stream),
                      static_cast<const T*>(x.data()),
                      static_cast<const T*>(x.data()) + cute::size(x.layout),
                      static_cast<const T*>(y.data()),
                      static_cast<T*>(error.data()), squareDiff<T>{});
  };
  autoDispatch(error.dtype, f);
}

void backwardKernel(cudaStream_t stream, Tensor1D& dx, const Tensor1D& x,
                    const Tensor1D& y) {
  const auto size = cute::size(x.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min<decltype(size)>(128, size));
    dim3 grid(util::ceilDiv(size, std::min<decltype(size)>(128, size)));
    grad<<<grid, block, 0, stream>>>(static_cast<T*>(dx.data()),
                                     static_cast<const T*>(x.data()),
                                     static_cast<const T*>(y.data()), size);
  };
  autoDispatch(x.dtype, f);
}
}  // namespace dllm::compute::Mse
