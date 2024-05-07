#include <curand_kernel.h>

#include "compute/random.h"
#include "util.h"

namespace dllm::compute::Random {
namespace {
template <typename T>
__global__ void kaimingNorm(T* y, curandState_t curandState, double stddev,
                            std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  skipahead(tid, &curandState);
  y[tid] = static_cast<T>(curand_normal(&curandState) * stddev);
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

void kaimingNormKernel(const ContextCompute* context, Tensor2D& y,
                       double stddev) {
  const auto size = cute::size(y.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min<decltype(size)>(128, size));
    dim3 grid(util::ceilDiv(size, std::min<decltype(size)>(128, size)));
    kaimingNorm<<<grid, block, 0, context->cudaStream>>>(
        static_cast<T*>(y.data()), context->curandState, stddev, size);
  };
  autoDispatch(y.dtype, f);
  skipahead(size, context->curandState, context->curandStateMutex);
}
}  // namespace dllm::compute::Random
