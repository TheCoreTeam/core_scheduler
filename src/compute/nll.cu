#include "compute/nll.h"
#include "logger.h"
#include "util.h"

namespace std {
__device__ auto log(const nv_half &x) { return hlog(x); }
__device__ auto log(const nv_bfloat16 &x) { return hlog(x); }
}  // namespace std

namespace dllm::compute::NLL {
namespace {
template <typename DataType, typename IndexType>
__global__ void nll(DataType *__restrict loss, const DataType *__restrict input,
                    const IndexType *__restrict target, const std::size_t ld,
                    const std::size_t n) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const auto targetIdx = target[tid];
  const auto prob = input[targetIdx + tid * ld];
  loss[targetIdx] = -std::log(prob);
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

void nllForwardKernel(cudaStream_t stream, Tensor1D &loss,
                      const Tensor2D &input, const Tensor2D &target) {
  if (target.dtype != R_32I) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Only support int32 for the target");
  }
  const auto size = cute::size(loss.layout);
  const auto ld = cute::shape<1>(input.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min<decltype(size)>(128, size));
    dim3 grid(util::ceilDiv(size, std::min<decltype(size)>(128, size)));
    nll<<<grid, block, 0, stream>>>(
        static_cast<T *>(loss.data()), static_cast<const T *>(input.data()),
        static_cast<const int *>(target.data()), ld, size);
  };
  autoDispatch(input.dtype, f);
}
}  // namespace dllm::compute::NLL