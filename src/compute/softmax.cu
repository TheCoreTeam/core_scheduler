#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <complex>

#include "compute/softmax.h"
#include "util.h"

namespace std {
__device__ auto max(const nv_half& a, const nv_half& b) { return __hmax(a, b); }
__device__ auto max(const nv_bfloat16& a, const nv_bfloat16& b) {
  return __hmax(a, b);
}
__device__ nv_half exp(const nv_half& x) {
  return std::exp(static_cast<float>(x));
}
__device__ nv_bfloat16 exp(const nv_bfloat16& x) {
  return std::exp(static_cast<float>(x));
}
};  // namespace std

namespace dllm::compute::SoftMax {
namespace {
template <typename floatX>
__global__ void softmax_forward_kernel5(
    floatX* __restrict out, floatX inv_temperature,
    const floatX* __restrict inp, const std::size_t N, const std::size_t T,
    const std::size_t ldO, const std::size_t ldI) {
  // inp, out shape: (N, T, T), where N = B * NH
  // fuses the multiplication by scale inside attention
  // directly autoregressive, so we only compute the lower triangular part
  // uses the online softmax algorithm
  assert(T % 4 == 0);
  constexpr int warp_size = 32;
  const auto lane_id = threadIdx.x % warp_size;
  const auto warp_id = threadIdx.x / warp_size;
  const auto num_warps = blockDim.x / warp_size;

  // micro-optimization: we iterate backwards so that
  // after the softmax backward operation completes, the cache retains the
  // part of the matrix close to the upper left corner, which benefits the
  // matmul operation that immediately follows.
  // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); //
  // forward order
  const auto idx =
      (gridDim.x - blockIdx.x - 1) * num_warps + warp_id;  // backward order
  if (idx >= N * T) {
    return;
  }
  const auto own_pos = idx % T;
  const auto pos_by_4 = own_pos / 4;

  // one row of inp, i.e. inp[idx, :] of shape (T,)
  const floatX* x = inp + idx * ldI;

  // not INF, so we don't get NaNs accidentally when subtracting two values.
  const floatX flt_max =
      340282346638528859811704183484516925440.0f;  // to avoid including float.h
  floatX maxval = -flt_max;
  floatX sumval = 0.0f;

  auto x_aligned = reinterpret_cast<const floatX* __restrict>(
      __builtin_assume_aligned(x, 16));
  for (auto i = lane_id; i < pos_by_4; i += warp_size) {
    floatX regarray[4];
    for (int k = 0; k < 4; ++k) {
      regarray[k] = x_aligned[4 * i + k];
    }
    floatX old_maxval = maxval;
    for (auto& k : regarray) {
      maxval = std::max(maxval, k);
    }
    sumval *= expf(inv_temperature * (old_maxval - maxval));
    for (auto& k : regarray) {
      sumval += std::exp(inv_temperature * (k - maxval));
    }
  }

  if (4 * pos_by_4 + lane_id <= own_pos) {
    floatX old_maxval = maxval;
    maxval = std::max(maxval, x[4 * pos_by_4 + lane_id]);
    sumval *= std::exp(inv_temperature * (old_maxval - maxval));
    sumval += std::exp(inv_temperature * (x[4 * pos_by_4 + lane_id] - maxval));
  }

  auto warpReduceMax = [] __device__(auto val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      val = std::max(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
  };

  auto global_maxval = warpReduceMax(maxval);
  sumval *= std::exp(inv_temperature * (maxval - global_maxval));

  auto warpReduceSum = [] __device__(auto val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
  };

  floatX sum = warpReduceSum(sumval);
  floatX norm = static_cast<floatX>(1.f) / sum;

  // divide the whole row by the sum
  for (auto i = lane_id; i <= own_pos; i += warp_size) {
    // recalculation is faster than doing the round-trip through memory.
    floatX ev = std::exp(inv_temperature * (__ldcs(x + i) - global_maxval));
    __stcs(out + idx * ldO + i, ev * norm);
  }
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

void forwardKernel(cudaStream_t stream, Tensor2D& output, const Tensor2D& input,
                   const double scale) {
  const auto size = cute::size(output.layout);
  const auto rows = cute::shape<0>(output.layout);
  const auto cols = cute::shape<1>(output.layout);
  const auto ld = cute::stride<0>(input.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    const auto blockSize = std::min<decltype(rows)>(256, size);
    const auto gridSize = util::ceilDiv(size, blockSize);
    softmax_forward_kernel5<T><<<gridSize, blockSize, 0, stream>>>(
        static_cast<T*>(output.data()), scale,
        static_cast<const T*>(input.data()), rows, cols,
        cute::stride<0>(output.layout), cute::stride<0>(input.layout));
  };
  autoDispatch(input.dtype, f);
}
}  // namespace dllm::compute::SoftMax
