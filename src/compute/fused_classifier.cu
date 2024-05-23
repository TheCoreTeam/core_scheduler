#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <nvtx3/nvToolsExt.h>

#include <string>

#include "compute/fused_classifier.h"
#include "util.h"

namespace dllm::compute::FusedClassifier {
namespace {
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
  static constexpr const long size = sizeof(int4) / sizeof(ElementType);
  ElementType payload[size];
};

// load a Packed128 from an aligned memory address
template <class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
  return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template <class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
  return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template <class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
  *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template <class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
  __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but
// bypassing L1
template <class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
  __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

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

template <typename T>
struct UpCast {
  using type = T;
};

template <>
struct UpCast<nv_half> {
  using type = float;
};

template <>
struct UpCast<nv_bfloat16> {
  using type = float;
};

template <typename floatX>
__device__ floatX warpReduceMax(floatX val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val = std::max(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

template <typename floatX>
__device__ floatX warpReduceSum(floatX val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename floatX>
__device__ auto prepare_softmax_blockwide3(const int idx, const floatX* inp,
                                           const long V, const long P) {
  using x128 = Packed128<floatX>;
  using UpperType = typename UpCast<floatX>::type;
  // same but not float4
  // one row of inp, i.e. inp[idx, :] of shape (V,)

  const floatX* x = inp + idx * P;
  UpperType thread_maxval = -INFINITY;
  UpperType thread_sumval = 0.0f;
  int i = (V + x128::size - 1) / x128::size + threadIdx.x - blockDim.x;

  // special-case loop to handle the unaligned elements at the end of the array
  // this lets us skip the bounds check in the main loop below, which improves
  // performance
  // printf("%d : %d\n", int((i + 1) * x128::size), int(V));
  while ((i + 1) * x128::size > V) {
#pragma unroll
    for (int k = 0; k < x128::size; ++k) {
      if (i * x128::size + k >= V) {
        break;  // bounds checking against real V (rather than padded P)
      }
      UpperType v = x[i * x128::size + k];
      UpperType old_maxval = thread_maxval;
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
    for (int k = 0; k < x128::size; ++k) {
      UpperType v = packed_x[k];
      UpperType old_maxval = thread_maxval;
      thread_maxval = std::max(thread_maxval, v);
      thread_sumval *= std::exp((old_maxval - thread_maxval));
      thread_sumval += std::exp(v - thread_maxval);
    }
  }

  // Block Max Reduction -> Maths -> Block Sum Reduction
  UpperType block_maxval =
      blockReduce(warpReduceMax<UpperType>, thread_maxval, true, -INFINITY);
  thread_sumval *= std::exp(thread_maxval - block_maxval);
  UpperType block_sumval = blockReduce(warpReduceSum<UpperType>, thread_sumval);

  // return the softmax parameters
  return SoftmaxParams<UpperType>{1.f / block_sumval, block_maxval};
}

// From LLM.c
// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder"
// parts
template <typename floatX, bool WriteLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_kernel5(floatX* logits /* B x T x Vp */,
                             floatX* losses /* B x T */, floatX* probs,
                             const floatX _dloss,
                             const int* targets /* B x T */, const long V,
                             const long P) {
  using x128 = Packed128<floatX>;
  using UpperType = typename UpCast<floatX>::type;
  UpperType dloss = _dloss;
  const auto idx =
      gridDim.x -
      (blockIdx.x + 1);  // reverse order for cache hits on matmul data
  const auto ix = targets[idx];

  // softmax (reading B * T * V, same logits read again below, hopefully still
  // in cache)
  auto sp = prepare_softmax_blockwide3<floatX>(idx, logits, V, P);

  // calculate the probability needed for the loss and update (single-threaded)
  if (threadIdx.x == 0) {
    UpperType prob =
        std::exp(static_cast<UpperType>(logits[idx * P + ix]) - sp.Offset) *
        sp.Scale;
    losses[idx] = -std::log(prob);
  }

  // calculate the gradients directly, saves bandwidth from probs during
  // training but also supports writing probs for inference-only and debugging
  const floatX* logits_vec = logits + idx * P;
  for (auto i = threadIdx.x; i < V / x128::size; i += blockDim.x) {
    // this is the 2nd read of logits after the one in prepare_softmax2
    // it will be overwritten by the logits gradients which is when we reduce
    // cache persistence
    x128 packed_logits_vec =
        load128(logits_vec + i * x128::size);  // rely on cs of store128cs
    x128 packed_probs;
#pragma unroll
    for (int k = 0; k < x128::size; ++k) {
      const auto element = i * x128::size + k;
      UpperType prob =
          std::exp(static_cast<UpperType>(packed_logits_vec[k]) - sp.Offset) *
          sp.Scale;
      packed_probs[k] = prob;
      UpperType indicator = (element == ix) ? 1.0f : 0.0f;
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
      V / x128::size * x128::size;  // round down to multiple of x128::size
  for (auto i = threadIdx.x + unaligned_start; i < V; i += blockDim.x) {
    UpperType prob =
        std::exp(static_cast<UpperType>(logits_vec[i]) - sp.Offset) * sp.Scale;
    // printf("tid: %d, value: %f\n", int(threadIdx.x + blockIdx.x *
    // blockDim.x), float(prob));
    UpperType indicator = (i == ix) ? 1.0f : 0.0f;
    UpperType dlogit = (prob - indicator) * dloss;
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

template <typename Fn>
__inline__ __attribute__((always_inline)) void autoDispatch(const Dtype dtype,
                                                            Fn&& fn) {
  switch (dtype) {
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

// replaces logits with logit gradients
void fused_classifier(cudaStream_t stream,
                      const std::shared_ptr<Tensor3D>& logits,
                      const std::shared_ptr<Tensor2D>& losses,
                      const double dloss,
                      const std::shared_ptr<const Tensor2D>& targets,
                      const long B, const long T, const long V, const long P) {
  NVTX_RANGE_FN();
  auto f = [&](auto dummy) {
    using Dtype = std::decay_t<decltype(dummy)>;
    const int block_size = 1024;
    const auto N = B * T;
    const auto grid_size = N;
    fused_classifier_kernel5<Dtype><<<grid_size, block_size, 0, stream>>>(
        static_cast<Dtype*>(logits->data()),
        static_cast<Dtype*>(losses->data()), nullptr, dloss,
        static_cast<const int*>(targets->data()), V, P);
  };
  autoDispatch(logits->dtype, f);
}
}  // namespace

TaskCompute call(const std::shared_ptr<Tensor3D>& logits,
                 const std::shared_ptr<Tensor2D>& losses,
                 const std::shared_ptr<const Tensor2D>& targets) {
  if (cute::shape<0>(logits->layout) != cute::shape<0>(losses->layout) ||
      cute::shape<0>(logits->layout) != cute::shape<0>(targets->layout) ||
      cute::shape<1>(logits->layout) != cute::shape<1>(losses->layout) ||
      cute::shape<1>(logits->layout) != cute::shape<1>(targets->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  if (targets->dtype != R_32I) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Only support int32 for the target now");
  }
  if (logits->dtype == R_64F || losses->dtype == R_64F) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "FP64 is not supported by fused classifier");
  }
  auto task = TaskCompute{
      [logitsFuture = *logits->future, lossesFuture = *losses->future,
       targetsFuture = targets->future->wFuture, logits = logits,
       losses = losses,
       targets = targets](const ContextCompute* context) mutable {
        {
          const auto B = cute::shape<0>(logits->layout);
          const auto T = cute::shape<1>(logits->layout);
          const auto V = cute::shape<2>(logits->layout);
          const auto P = cute::stride<1>(logits->layout);
          util::FutureGuard logitsRGuard{logitsFuture.rFuture};
          util::FutureGuard logitsWGuard{logitsFuture.wFuture};
          util::FutureGuard lossesRGuard{lossesFuture.rFuture};
          util::FutureGuard lossesWGuard{lossesFuture.wFuture};
          util::FutureGuard targetsWGuard{targetsFuture};
          fused_classifier(context->cudaStream, logits, losses, 1. / (B * T),
                           targets, B, T, V, P);
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }
        logits.reset();
        losses.reset();
        targets.reset();
      }};

  const TaskFuture future = task.get_future();
  logits->future->wFuture = future;
  losses->future->wFuture = future;
  targets->future->rFuture = future;
  return task;
}
}  // namespace dllm::compute::FusedClassifier
