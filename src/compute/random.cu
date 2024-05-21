#include <curand_kernel.h>

#include "compute/random.h"
#include "random/random_internal.h"
#include "util.h"

namespace dllm::compute::Random {
namespace {
template <typename T>
__global__ void gaussian(T *y, const unsigned long curandSeed,
                         const unsigned long curandOffset, std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  curandStatePhilox4_32_10_t state;
  curand_init(curandSeed, tid, curandOffset, &state);
  y[tid] = static_cast<T>(curand_normal(&state));
}

template <typename T>
__global__ void uniform(T *y, const unsigned long curandSeed,
                        const unsigned long curandOffset, std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  curandStatePhilox4_32_10_t state;
  curand_init(curandSeed, tid, curandOffset, &state);
  y[tid] = static_cast<T>(curand_uniform(&state));
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

void gaussianKernel(const ContextCompute *context, Tensor1D &tensor) {
  const auto size = cute::size(tensor.layout);
  auto &[seed, offset] = random::getRandomState();
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min<decltype(size)>(128, size));
    dim3 grid(util::ceilDiv(size, std::min<decltype(size)>(128, size)));
    gaussian<<<grid, block, 0, context->cudaStream>>>(
        static_cast<T *>(tensor.data()), seed, offset.fetch_add(size), size);
  };
  autoDispatch(tensor.dtype, f);
}

void uniformKernel(const ContextCompute *context, Tensor1D &tensor) {
  const auto size = cute::size(tensor.layout);
  auto &[seed, offset] = random::getRandomState();
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min<decltype(size)>(128, size));
    dim3 grid(util::ceilDiv(size, std::min<decltype(size)>(128, size)));
    uniform<<<grid, block, 0, context->cudaStream>>>(
        static_cast<T *>(tensor.data()), seed, offset.fetch_add(size), size);
  };
  autoDispatch(tensor.dtype, f);
}
}  // namespace dllm::compute::Random
