#include <cuda_fp16.h>
//#include <sm_32_intrinsics.h>
#include <curand_kernel.h>
#include <sm_32_intrinsics.h>
#include <math_constants.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda_profiler_api.h>
#include "compute/embedding.h"
#include "util.h"

// Specific configurations based on the enabled precision
//#if defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#define PRECISION_MODE PRECISION_FP32
#ifdef MULTI_GPU
const ncclDataType_t ncclFloatX = ncclFloat;
#endif

//// use fp16 (note: this may require gradient scaler, currently not implemented!)
//#elif defined(ENABLE_FP16)
//typedef half floatX;
//#define CUBLAS_LOWP CUDA_R_16F
//#define PRECISION_MODE PRECISION_FP16
//#ifdef MULTI_GPU
//const ncclDataType_t ncclFloatX = ncclHalf;
//#endif
//
//#else // Default to bfloat16
//typedef __nv_bfloat16 floatX;
//#define CUBLAS_LOWP CUDA_R_16BF
//#define PRECISION_MODE PRECISION_BF16
//#ifdef MULTI_GPU
//const ncclDataType_t ncclFloatX = ncclBfloat16;
//#endif
//#endif

//#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
//__device__ floatX __ldcs(const floatX* address) {
//  unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
//  return __nv_bfloat16_raw{bf};
//}
//
//__device__ void __stcs(floatX* address, floatX value) {
//  __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
//}
//#endif

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace dllm::compute::embedding {
namespace {
//template <typename T>
//__global__ void embedding_forward(T* output, const int* __restrict__ input, const T* __restrict__ weight,
//                          std::size_t B, std::size_t input_length, std::size_t embedding_dim) {
//  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//  int N = B*input_length*embedding_dim;
//  if (tid >= N) {
//    return;
//  }
//
//  int bt = tid / embedding_dim;
//  int b = bt / input_length;
//  int t = bt % input_length;
//  int c = tid % embedding_dim;
//
//  int ix = input[b * input_length + t];
//  int input_idx = input[tid];
//  output[b * input_length * embedding_dim + t * embedding_dim + c] = weight[ix * embedding_dim + c];
//}

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

//__device__ floatX __ldcs(const floatX* address) {
//  unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
//  return __nv_bfloat16_raw{bf};
//}


template<class ElementType>
struct alignas(16) Packed128 {
  Packed128() = default;
  __device__ explicit Packed128(int4 bits) {
    static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
    memcpy(&payload, &bits, sizeof(bits));
  }
  __device__ ElementType& operator[](int index) {
    return payload[index];
  }
  __device__ const ElementType& operator[](int index) const {
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
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
  return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
  return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
  *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
  __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
  __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}



template <typename floatX>
__global__ void embedding_forward(floatX* out,
                                        const int* inp, const floatX* wte, const floatX* wpe,
                                        int B, int T, int C) {

  // short-form typedefs
  typedef Packed128<float> f128;
  typedef Packed128<floatX> x128;

  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
  int N = B * T * C;
  if (idx >= N) { return; }

  int bt = idx / C;
  int b = bt / T;
  int t = bt % T;
  int c = idx % C;

  int ix = inp[b * T + t];

  floatX* out_btc = out + b * T * C + t * C + c;
  const floatX* wte_ix = wte + ix * C + c;
  const floatX* wpe_tc = wpe + t * C + c;

  x128 packed_out;
  x128 wte128 = load128cs(wte_ix);
  x128 wpe128 = load128cs(wpe_tc);
  for (int k = 0; k < x128::size; k++) {
    packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
  }
  store128(out_btc, packed_out);
}


//template <typename T>
//__global__ void embedding_backward(const T* grad_output, int* grad_input, T* grad_weight,
//                                   std::size_t B, std::size_t input_length, std::size_t embedding_dim) {
//  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
//  int N = B * input_length * embedding_dim;
//  if (tid >= N) {
//    return;
//  }
//
//  int bt = tid / embedding_dim;
//  int b = bt / input_length;
//  int t = bt % input_length;
//  int c = tid % embedding_dim;
//
//  int ix = grad_input[b * input_length + t];
//
//  atomicAdd(&grad_weight[ix * embedding_dim + c], grad_output[b * input_length * embedding_dim + t * embedding_dim + c]);
//
//}

__device__ __host__ unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// todo - possibly overkill and we don't need such high quality random numbers? (tbd)
// http://eiserloh.net/noise/SquirrelNoise5.hpp
__device__ __host__ constexpr unsigned int SquirrelNoise5(int positionX, unsigned int seed)
{
  constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
  constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
  constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
  constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
  constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
  unsigned int mangledBits = (unsigned int) positionX;
  mangledBits *= SQ5_BIT_NOISE1;
  mangledBits += seed;
  mangledBits ^= (mangledBits >> 9);
  mangledBits += SQ5_BIT_NOISE2;
  mangledBits ^= (mangledBits >> 11);
  mangledBits *= SQ5_BIT_NOISE3;
  mangledBits ^= (mangledBits >> 13);
  mangledBits += SQ5_BIT_NOISE4;
  mangledBits ^= (mangledBits >> 15);
  mangledBits *= SQ5_BIT_NOISE5;
  mangledBits ^= (mangledBits >> 17);
  return mangledBits;
}

__device__ __host__ constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
  constexpr int PRIME_NUMBER = 198491317; // Large prime number with non-boring bits
  return SquirrelNoise5(indexX + (PRIME_NUMBER * indexY), seed);
}

// stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
__device__ __forceinline__ void stochastic_rounding(float in, __nv_bfloat16 *out, unsigned int seed) {
  // todo - is this stochastic rounding *too good*? can we cut any corners?
  unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x, seed);
  unsigned int threshold = random & 0xFFFF;
  unsigned int float_bits = __float_as_uint(in);
  unsigned int rounded_bits = float_bits & 0x0000FFFF;
  float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
  *out = __float2bfloat16_rn(__uint_as_float(float_bits));
}
__device__ __forceinline__ void stochastic_rounding(float in, half *out, unsigned int random) {
  *out = (float)in; // todo - implement this...
}
__device__ __forceinline__ void stochastic_rounding(float in, float *out, unsigned int random) {
  *out = in; // dummy function for when floatX is float (FP32 mode)
}

template <typename T>
__device__ void atomicStochasticAdd(T* address, float val0, float val1, unsigned int seed) {
  static_assert(sizeof(T) == 2, "Only 16-bit atomicStochasticAdd supported.");
  float2 val = make_float2(val0, val1);
  unsigned int* address_as_uint = (unsigned int*)address;
  unsigned int old = *address_as_uint, assumed;
  unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x, seed);
  do {
    assumed = old;
    float2 new_fp32 = make_float2((float)(reinterpret_cast<T*>(&old)[0]) + val.x,
                                  (float)(reinterpret_cast<T*>(&old)[1]) + val.y);
    T new_rounded[2];
    stochastic_rounding(new_fp32.x, &new_rounded[0], random);
    stochastic_rounding(new_fp32.y, &new_rounded[1], random >> 16);
    old = atomicCAS(address_as_uint, assumed, *(unsigned int*)&new_rounded);
  } while (assumed != old);
}

__device__ void atomicStochasticAdd(float* address, float val0, float val1, unsigned int seed) {
  atomicAdd(address, val0);
  atomicAdd(address + 1 , val1);
}

template <typename floatX>
__global__ void embedding_backward(floatX* dwte, floatX* dwpe,
                                        const floatX* dout, const int* inp,
                                        int B, int T, int C, unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = B * T * C;
  idx *= 2; // 2 elements per thread
  if (idx >= N) { return; }

  int bt = idx / C;
  int b = bt / T;
  int t = bt % T;
  int c = idx % C;

  int ix = inp[b * T + t];

  const floatX* dout_btc = dout + b * T * C + t * C + c;
  floatX* dwte_ix = dwte + ix * C + c;
  floatX* dwpe_tc = dwpe + t * C + c;

  float2 dout_data = make_float2(dout_btc[0], dout_btc[1]);
  atomicStochasticAdd(dwte_ix, dout_data.x, dout_data.y, seed);
  atomicStochasticAdd(dwpe_tc, dout_data.x, dout_data.y, seed ^ 0xFFFFFFFF);
}

template <typename Fn>
__inline__ __attribute__((always_inline)) void autoDispatch(Dtype dtype,
                                                            Fn&& fn) {
  switch (dtype) {
//    case R_64F:
//      fn(double{0});
//      return;
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

void forwardKernel(cudaStream_t cudaStream, Tensor3D& output,
                   const Tensor2D& input, const Tensor2D& wte, const Tensor2D& wpe) {
  const auto B = cute::shape<0>(input.layout);
  const auto L =  cute::shape<1>(input.layout);
  const auto vocab = cute::shape<0>(wte.layout);
  const auto embedding_dim = cute::shape<1>(wte.layout);
  const auto N =B * L * embedding_dim;
  auto f = [&](auto dummy) {
    using floatX = std::remove_const_t<std::decay_t<decltype(dummy)>>;
//    dim3 block(std::min<decltype(N)>(128, N));
//    dim3 grid(util::ceilDiv(N, std::min<decltype(N)>(128, N)));
    typedef Packed128<floatX> x128;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
    embedding_forward<<<grid_size, block_size, 0, cudaStream>>>(static_cast<floatX *>(output.data()),
                                         static_cast<const int*>(input.data()),static_cast<const floatX*>(wte.data()),
                                                      static_cast<const floatX*>(wpe.data()),
                                                      B, L, embedding_dim);
  };
  autoDispatch(output.dtype, f);
}

void backwardKernel(cudaStream_t cudaStream, const Tensor3D& grad_output,
                   Tensor2D& grad_input, Tensor2D& grad_wte, Tensor2D& grad_wpe
                    ) {
  const auto B = cute::shape<0>(grad_output.layout);
  const auto L =  cute::shape<1>(grad_output.layout);
  const auto vocab = cute::shape<0>(grad_wte.layout);
  const auto embedding_dim = cute::shape<1>(grad_wte.layout);
  const auto N =B * L * embedding_dim;
  unsigned long long rng_state = 13371337;

  auto f = [&](auto dummy) {
    using floatX = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(256);
    dim3 grid(CEIL_DIV(N, block.x * 2));
    embedding_backward<<<grid, block, 0, cudaStream>>>(static_cast<floatX*>(grad_wte.data()),
                                                       static_cast<floatX*>(grad_wpe.data()),
                                                       static_cast<const floatX*>(grad_output.data()),
                                                      static_cast<int*>(grad_input.data()),
                                                       B, L, embedding_dim, random_u32(&rng_state));
  };
  autoDispatch(grad_wte.dtype, f);
}
}  // namespace dllm::compute::GeLU8
