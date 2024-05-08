#include "tensor.h"
#include "util.h"

namespace dllm::compute {
template <typename Element>
__global__ void relu(const Element* __restrict__ input, Element* __restrict__ output, const size_t size) {

  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < size) {
    output[tid] = std::max<Element>(0, input[tid]);
  }

}

namespace {
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

void reluKernel(cudaStream_t stream, const Tensor1D& input, Tensor1D& output) {
  const auto size = cute::size(input.layout);
  const dim3 block(std::min(128, size));
  const dim3 grid(util::ceilDiv(size, std::min(128, size)));

  auto f = [&](auto dummy) {
    using Element = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    relu<Element><<<grid, block, 0, stream>>>(static_cast<const Element*>(input.data()),
                                              static_cast<Element*>(output.data()), size);
  };

  autoDispatch(input.dtype, f);
}
} // namespace dllm::compute
