#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "compute/gelu.h"
#include "util.h"

namespace dllm::compute{
namespace {
template <typename T>
__global__ void GeLU(T* output, const T* input, std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  int a=2;
//  output[tid] = 1;
  output[tid] = static_cast<T>(0.5) * input[tid] * (static_cast<T>(1) + static_cast<T>(tanhf(static_cast<T>(CUDART_SQRT_2OPI) * (input[tid] + static_cast<T>(0.044715) * input[tid] * input[tid] * input[tid]))));

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
