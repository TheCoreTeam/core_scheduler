#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <nvtx3/nvToolsExt.h>

#include <format>
#include <string>

#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

template <class ElementType>
struct alignas(16) Packed128 {
  Packed128() = default;
  __device__ explicit Packed128(const int4 bits) {
    static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
    memcpy(&payload, &bits, sizeof(bits));
  }
  __device__ ElementType& operator[](const int index) { return payload[index]; }
  __device__ const ElementType& operator[](const int index) const {
    return payload[index];
  }
  __device__ int4 get_bits() const {
    int4 bits;
    static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
    memcpy(&bits, &payload, sizeof(bits));
    return bits;
  }
  static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
  ElementType payload[size];
};

template <typename floatX>
struct SoftmaxParams {
  floatX Scale;
  floatX Offset;
};

template <typename Fn, typename floatX>
__device__ floatX blockReduce(Fn&& fn, const floatX val,
                              const bool final_sync = false,
                              const floatX out_of_bounds = 0.0f) {
  // two reductions of up to 1024 threads:
  // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp
  // (shuffle)
  __shared__ floatX shared_val[32];
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  const int num_warps = blockDim.x / 32;

  auto warp_val = fn(val);
  if (lane_id == 0) {
    shared_val[warp_id] = warp_val;
  }
  __syncthreads();
  warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
  auto block_val = fn(warp_val);

  if (final_sync) {
    __syncthreads();  // only needed in loops when effectively reusing shared
                      // memory etc.
  }
  return block_val;
}

template <typename floatX>
__device__ float warpReduceMax(floatX val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val = std::max(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

template <typename floatX>
__device__ float warpReduceSum(floatX val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename floatX>
__device__ SoftmaxParams<floatX> prepare_softmax_blockwide3(const int idx,
                                                            const floatX* inp,
                                                            const int V,
                                                            const int P) {
  using x128 = Packed128<floatX>;
  // same but not float4
  // one row of inp, i.e. inp[idx, :] of shape (V,)

  const floatX* x = inp + idx * P;
  floatX thread_maxval = -INFINITY;
  floatX thread_sumval = 0.0f;
  int i = (V + x128::size - 1) / x128::size + threadIdx.x - blockDim.x;

  // special-case loop to handle the unaligned elements at the end of the array
  // this lets us skip the bounds check in the main loop below, which improves
  // performance
  while ((i + 1) * x128::size > V) {
#pragma unroll
    for (int k = 0; k < x128::size; ++k) {
      if (i * x128::size + k >= V) {
        break;  // bounds checking against real V (rather than padded P)
      }
      const auto v = x[i * x128::size + k];
      const auto old_maxval = thread_maxval;
      thread_maxval = std::max(thread_maxval, v);
      thread_sumval *= std::exp((old_maxval - thread_maxval));
      thread_sumval += std::exp(v - thread_maxval);
    }
    i -= blockDim.x;
  }

  // main loop for the bulk of the iterations (no bounds checking required!)
  for (; i >= 0; i -= blockDim.x) {
    x128 packed_x = load128(
        x +
        i * x128::size);  // load and keep in cache until fused_classifier loop
#pragma unroll
    for (int k = 0; k < x128::size; ++k) {
      const auto v = packed_x[k];
      const auto old_maxval = thread_maxval;
      thread_maxval = std::max(thread_maxval, v);
      thread_sumval *= std::exp((old_maxval - thread_maxval));
      thread_sumval += std::exp(v - thread_maxval);
    }
  }

  // Block Max Reduction -> Maths -> Block Sum Reduction
  auto block_maxval =
      blockReduce(warpReduceMax<floatX>, thread_maxval, false, -INFINITY);
  thread_sumval *= std::exp(thread_maxval - block_maxval);
  auto block_sumval = blockReduce(warpReduceSum<floatX>, thread_sumval);

  // return the softmax parameters
  return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// From LLM.c
// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder"
// parts
template <typename floatX, bool WriteLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_kernel5(const floatX* logits, floatX* losses,
                             floatX* probs, const float dloss,
                             const int* targets, const int B, const int T,
                             const int V, const int P) {
  using x128 = Packed128<floatX>;
  int idx = gridDim.x -
            (blockIdx.x + 1);  // reverse order for cache hits on matmul data
  int ix = targets[idx];

  // softmax (reading B * T * V, same logits read again below, hopefully still
  // in cache)
  SoftmaxParams<floatX> sp = prepare_softmax_blockwide3(idx, logits, V, P);

  // calculate the probability needed for the loss and update (single-threaded)
  if (threadIdx.x == 0) {
    const auto prob = std::exp(logits[idx * P + ix] - sp.Offset) * sp.Scale;
    losses[idx] = -std::log(prob);
  }

  // calculate the gradients directly, saves bandwidth from probs during
  // training but also supports writing probs for inference-only and debugging
  const floatX* logits_vec = logits + idx * P;
  for (int i = threadIdx.x; i < V / x128::size; i += blockDim.x) {
    // this is the 2nd read of logits after the one in prepare_softmax2
    // it will be overwritten by the logits gradients which is when we reduce
    // cache persistence
    x128 packed_logits_vec =
        load128(logits_vec + i * x128::size);  // rely on cs of store128cs
    x128 packed_probs;
#pragma unroll
    for (int k = 0; k < x128::size; ++k) {
      const auto element = i * x128::size + k;
      floatX prob = std::exp(packed_logits_vec[k] - sp.Offset) * sp.Scale;
      packed_probs[k] = prob;
      const auto indicator = (element == ix) ? 1.0f : 0.0f;
      packed_logits_vec[k] = (prob - indicator) * dloss;
    }
    if (WriteLogits) {
      // reduce cache persistence for the overwritten logits
      // to maximise probability that logits remain in cache between
      // prepare_softmax and here
      store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
    }
    if (WriteProbs) {
      store128(probs + idx * P + i * x128::size, packed_probs);
    }
  }

  // handle remaining elements after the last multiple of x128::size
  // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
  const int unaligned_start =
      V & ~(x128::size - 1);  // round down to multiple of x128::size
  for (int i = threadIdx.x + unaligned_start; i < V; ++i) {
    const auto prob = std::exp(logits_vec[i] - sp.Offset) * sp.Scale;
    const auto indicator = i == ix ? 1.0f : 0.0f;
    const auto dlogit = (prob - indicator) * dloss;
    if (WriteLogits) {
      __stcs(logits + idx * P + i, dlogit);
    }
    if (WriteProbs) {
      probs[idx * P + i] = prob;
    }
  }
}

class NvtxRange {
 public:
  NvtxRange(const char* s) { nvtxRangePush(s); }
  NvtxRange(const std::string& base_str, int number) {
    const std::string range_string = std::format("{} {}", base_str, number);
    nvtxRangePush(range_string.c_str());
  }
  ~NvtxRange() { nvtxRangePop(); }
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// replaces logits with logit gradients
template <typename Type>
void fused_classifier(Type* logits, Type* losses, const Type dloss,
                      const int* targets, const int B, const int T, const int V,
                      const int P) {
  NVTX_RANGE_FN();
  const int block_size = 1024;
  const int N = B * T;
  const int grid_size = N;
  fused_classifier_kernel5<<<grid_size, block_size, 512>>>(
      logits, losses, nullptr, dloss, targets, B, T, V, P);
}