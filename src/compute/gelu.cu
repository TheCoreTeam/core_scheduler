#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include "compute/gelu.h"
#include "util.h"
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <class ElementType>
struct alignas(16) Packed128 {
  Packed128() = default;
  __device__ explicit Packed128(int4 bits) {
    static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
    memcpy(&payload, &bits, sizeof(bits));
  }
  __device__ ElementType& operator[](int index) { return payload[index]; }
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

namespace dllm::compute::GeLU {
namespace {
template <typename T>
__global__ void GeLU_forward(T* __restrict__ output,
                             const T* __restrict__ input, std::size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }

  // For double use double erf, for single, half, bfloat16 use single efr
  constexpr auto useDouble = sizeof(T) > sizeof(float);
  using TargetType = std::conditional_t<useDouble, double, float>;
  TargetType inputElement = input[tid];

  // output[tid] =
  //     static_cast<TargetType>(0.5) * inputElement *
  //     (static_cast<TargetType>(1.) +
  //      erf(inputElement * static_cast<TargetType>(1 / CUDART_SQRT_HALF)));
  output[tid] = static_cast<TargetType>(0.5) * inputElement *
                (static_cast<TargetType>(1.) +
                 std::tanh(std::sqrt(2. / M_PI) *
                           (inputElement + 0.044715 * inputElement *
                                               inputElement * inputElement)));
}

template <typename T>
__device__ T phi(T x) {
  return std::exp(-0.5f * x * x) / std::sqrt(2.0f * M_PI);
}

template <typename T>
__device__ T Phi(T x) {
  return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

template <typename floatX>
__global__ void GeLU_backward(floatX* dinp, const floatX* inp,
                              const floatX* dout) {
  typedef Packed128<floatX> x128;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

  x128 packed_dinp;
  x128 packed_inp = load128cs(inp + idx);
  x128 packed_dout = load128cs(dout + idx);
  for (int k = 0; k < packed_inp.size; ++k) {
    float x = (float)packed_inp[k];
    // float cube = 0.044715f * x * x * x;
    // float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
    // float tanh_out = tanhf(tanh_arg);
    // float coshf_out = coshf(tanh_arg);
    // float sech_out = 1.0f / (coshf_out * coshf_out);
    // float local_grad =
    //     0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR
    //     *
    //                                    (1.0f + 3.0f * 0.044715f * x * x);
    float phi_x = phi(x);
    float Phi_x = Phi(x);
    float local_grad = Phi_x + x * phi_x;
    packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
  }
  store128(dinp + idx, packed_dinp);
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

void forwardKernel(cudaStream_t cudaStream, Tensor1D& output,
                   const Tensor1D& input) {
  const auto size = cute::size(input.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    dim3 block(std::min<decltype(size)>(128, size));
    dim3 grid(util::ceilDiv(size, std::min<decltype(size)>(128, size)));
    GeLU_forward<<<grid, block, 0, cudaStream>>>(
        static_cast<T*>(output.data()), static_cast<const T*>(input.data()),
        size);
  };
  autoDispatch(output.dtype, f);
}

void backwardKernel(cudaStream_t cudaStream, Tensor1D& dinput,
                    const Tensor1D& input, const Tensor1D& doutput) {
  const auto size = cute::size(input.layout);
  auto f = [&](auto dummy) {
    using T = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    typedef Packed128<T> x128;
    dim3 block(128);
    dim3 grid(CEIL_DIV(size, block.x * x128::size));
    GeLU_backward<<<grid, block, 0, cudaStream>>>(
        static_cast<T*>(dinput.data()), static_cast<const T*>(input.data()),
        static_cast<const T*>(doutput.data()));
  };
  autoDispatch(dinput.dtype, f);
}
}  // namespace dllm::compute::GeLU
