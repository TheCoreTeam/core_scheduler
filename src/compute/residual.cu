#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "tensor.h"
#include "util.h"

namespace dllm::compute::Residual {
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

template <typename T>
struct element_add {
  __host__ __device__ T operator()(const T& x, const T& y) const {
    return x + y;
  }
};

void forwardKernel(cudaStream_t stream, const Tensor3D& A, const Tensor3D& B,
                   Tensor3D& C) {
  const auto size = cute::size(A.layout);

  auto f = [&](auto dummy) {
    using Element = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    thrust::device_ptr<const Element> dev_ptr_A(
        static_cast<const Element*>(A.data()));
    thrust::device_ptr<const Element> dev_ptr_B(
        static_cast<const Element*>(B.data()));
    thrust::device_ptr<Element> dev_ptr_C(static_cast<Element*>(C.data()));
    thrust::transform(thrust::cuda::par.on(stream), dev_ptr_A, dev_ptr_A + size,
                      dev_ptr_B, dev_ptr_C, element_add<Element>());
  };

  autoDispatch(A.dtype, f);
}

void backwardKernel(cudaStream_t stream, const Tensor3D& grad_output,
                    Tensor3D& grad_A, Tensor3D& grad_B) {
  const auto size = cute::size(grad_output.layout);

  auto f = [&](auto dummy) {
    using Element = std::remove_const_t<std::decay_t<decltype(dummy)>>;
    thrust::device_ptr<const Element> dev_ptr_grad_output(
        static_cast<const Element*>(grad_output.data()));
    thrust::device_ptr<Element> dev_ptr_grad_A(
        static_cast<Element*>(grad_A.data()));
    thrust::device_ptr<Element> dev_ptr_grad_B(
        static_cast<Element*>(grad_B.data()));

    thrust::copy(thrust::cuda::par.on(stream), dev_ptr_grad_output,
                 dev_ptr_grad_output + size, dev_ptr_grad_A);
    thrust::copy(thrust::cuda::par.on(stream), dev_ptr_grad_output,
                 dev_ptr_grad_output + size, dev_ptr_grad_B);
  };

  autoDispatch(grad_output.dtype, f);
}

}  // namespace dllm::compute::Residual
