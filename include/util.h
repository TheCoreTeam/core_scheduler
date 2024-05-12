#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::util {
template <typename Future>
__inline__ __attribute__((always_inline)) void waitFutureIfValid(
    Future &&future) {
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
__inline__ __attribute__((always_inline)) Tensor<OutN> flatten_impl(
    const Tensor<InN> &tensor, std::true_type /* OutN != 1 */) {
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
__inline__ __attribute__((always_inline)) Tensor<OutN> flatten_impl(
    const Tensor<InN> &tensor, std::false_type /* OutN == 1 */) {
  return {tensor.data(),
          cute::make_layout(cute::shape(cute::size(tensor.layout)),
                            cute::make_stride(cute::_1{})),
          tensor.dtype, tensor.deviceType, tensor.future};
}

template <int OutN, int InN>
__inline__ __attribute__((always_inline)) Tensor<OutN> flatten(
    const Tensor<InN> &tensor) {
  static_assert(InN > 1 && OutN < InN);
  if constexpr (OutN != 1) {
    return flatten_impl<OutN>(tensor, std::true_type{});
  } else {
    return flatten_impl<OutN>(tensor, std::false_type{});
  }
}

template <int OutN, template <int> class T, int InN>
auto flatten_impl(const std::shared_ptr<T<InN>> &tensor,
                  std::true_type /* OutN != 1 */) {
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
auto flatten_impl(const std::shared_ptr<T<InN>> &tensor,
                  std::false_type /* OutN == 1 */) {
  return std::make_shared<T<OutN>>(
      tensor->data(),
      cute::make_layout(cute::make_shape(cute::size(tensor->layout)),
                        cute::make_stride(cute::_1{})),
      tensor->dtype, tensor->deviceType, tensor->future);
}

template <int OutN, template <int> class T, int InN>
auto flatten(const std::shared_ptr<T<InN>> &tensor) {
  static_assert(isTensor<T, InN>);
  static_assert(InN > 1 && OutN < InN);
  if constexpr (OutN != 1) {
    return flatten_impl<OutN>(tensor, std::true_type{});
  } else {
    return flatten_impl<OutN>(tensor, std::false_type{});
  }
}

// Binary hacking to improve performance - be careful
template <typename T>
__inline__ __attribute__((always_inline)) const std::shared_ptr<const T> &
toConstSharedPtr(const std::shared_ptr<T> &ptr) {
  static_assert(sizeof(const std::shared_ptr<T> &) ==
                sizeof(const std::shared_ptr<const T> &));
  return reinterpret_cast<const std::shared_ptr<const T> &>(ptr);
}

// Binary hacking to improve performance - be careful
template <typename T>
__inline__ __attribute__((always_inline)) std::shared_ptr<const T> &
toConstSharedPtr(std::shared_ptr<T> &ptr) {
  static_assert(sizeof(std::shared_ptr<T> &) ==
                sizeof(std::shared_ptr<const T> &));
  return reinterpret_cast<std::shared_ptr<const T> &>(ptr);
}
}  // namespace dllm::util
