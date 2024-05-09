#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "compute/gelu.h"
#include "util.h"

namespace dllm::compute{
namespace {
template <typename T>
__global__ void GeLU(T* __restrict__ output, const T* __restrict__ input, std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }

  constexpr auto inv_sqrt_2 = 0.7071067811865475;
  output[tid] = float{0.5} * static_cast<float>(input[tid]) * (float{1} + erf(static_cast<float>(input[tid]) * float{inv_sqrt_2}));

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

void GeLUKernel(cudaStream_t cudaStream, Tensor1D &output, const Tensor1D &input) {
  const auto size = cute::size(input.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min(128, size));
    dim3 grid(util::ceilDiv(size, std::min(128, size)));
    GeLU<<<grid, block, 0, cudaStream>>>(static_cast<T*>(output.data()),
                                         static_cast<const T*>(input.data()),
                                         size);
  };
  autoDispatch(output.dtype, f);
}
}  // namespace dllm::compute::Init
