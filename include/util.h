#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::util {
__inline__ __attribute__((always_inline)) void waitFutureIfValid(
    const std::shared_ptr<FutureCompute> &future) {
  if (future != nullptr && future->valid()) {
    future->wait();
  }
}

__inline__ __attribute__((always_inline)) void waitFutureIfValid(
    const FutureCompute &future) {
  if (future.valid()) {
    future.wait();
  }
}

constexpr __inline__ __attribute__((always_inline)) int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

constexpr __inline__ __attribute__((always_inline)) long ceilDiv(long a,
                                                                 long b) {
  return (a + b - 1) / b;
}

template <int OutN, int InN>
  requires(InN > 1 && OutN < InN && OutN != 1)
__inline__ __attribute__((always_inline)) Tensor<OutN> flatten(
    const Tensor<InN> &tensor) {
  return {tensor.data(),
          cute::make_layout(
              cute::make_layout(
                  cute::size(
                      cute::take<0, decltype(tensor.layout)::rank - (OutN - 1)>(
                          tensor.layout)),
                  cute::stride<decltype(tensor.layout)::rank - OutN>(
                      tensor.layout)),
              cute::layout<decltype(tensor.layout)::rank - (OutN - 1)>(
                  tensor.layout)),
          tensor.dtype, tensor.deviceType, tensor.future};
}

template <int OutN, int InN>
  requires(InN > 1 && OutN < InN && OutN == 1)
__inline__ __attribute__((always_inline)) Tensor<OutN> flatten(
    const Tensor<InN> &tensor) {
  return {tensor.data(),
          cute::make_layout(cute::shape(cute::size(tensor.layout)),
                            cute::make_stride(cute::_1{})),
          tensor.dtype, tensor.deviceType, tensor.future};
}

template <int OutN, template <int> class T, int InN>
  requires isTensor<T, InN> && (InN > 1) && (OutN < InN) && (OutN != 1)
__inline__ __attribute__((always_inline)) auto flatten(
    const std::shared_ptr<T<InN>> &tensor) {
  return std::make_shared<T<OutN>>(
      tensor->data(),
      cute::make_layout(
          cute::make_layout(
              cute::size(
                  cute::take<0, decltype(tensor->layout)::rank - (OutN - 1)>(
                      tensor->layout)),
              cute::stride<decltype(tensor->layout)::rank - OutN>(
                  tensor->layout)),
          cute::layout<decltype(tensor->layout)::rank - (OutN - 1)>(
              tensor->layout)),
      tensor->dtype, tensor->deviceType, tensor->future);
}

template <int OutN, template <int> class T, int InN>
  requires isTensor<T, InN> && (InN > 1) && (OutN < InN) && (OutN == 1)
__inline__ __attribute__((always_inline)) auto flatten(
    const std::shared_ptr<T<InN>> &tensor) {
  return std::make_shared<T<OutN>>(
      tensor->data(),
      cute::make_layout(cute::make_shape(cute::size(tensor->layout)),
                        cute::make_stride(cute::_1{})),
      tensor->dtype, tensor->deviceType, tensor->future);
}

template <typename T>
__inline__ __attribute__((always_inline)) std::shared_ptr<const T>
toConstSharedPtr(const std::shared_ptr<T> &ptr) {
  return std::shared_ptr<const T>(ptr);
}
}  // namespace dllm::util
