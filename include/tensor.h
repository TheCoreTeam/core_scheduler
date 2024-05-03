#pragma once
#include <cute/layout.hpp>
#include <future>
#include <memory>
#include <tuple>
#include <vector>

#include "device.h"
#include "dtype.h"
#include "task.h"

namespace dllm {

template <typename T, int N, template <typename...> class Template,
          typename... Args>
struct repeat_type {
  static_assert(N >= 0);
  using type = typename repeat_type<T, N - 1, Template, T, Args...>::type;
};

template <typename T, template <typename...> class Template, typename... Args>
struct repeat_type<T, 0, Template, Args...> {
  using type = Template<Args...>;
};

template <typename T, int N, template <typename...> class Template,
          typename... Args>
struct repeat_last_1_type {
  static_assert(N >= 0);
  using type =
      typename repeat_last_1_type<T, N - 1, Template, T, Args...>::type;
};

template <typename T, template <typename...> class Template, typename... Args>
struct repeat_last_1_type<T, 0, Template, Args...> {
  using type = Template<Args..., cute::_1>;
};

template <int N>
struct Tensor {
  using Shape = typename repeat_type<int, N, cute::Shape>::type;
  using Stride = typename repeat_last_1_type<int, N - 1, cute::Stride>::type;
  using Layout = cute::Layout<Shape, Stride>;

  Tensor(const Tensor &tensor)
      : data_{tensor.data_},
        layout{tensor.layout},
        dtype{tensor.dtype},
        deviceType{tensor.deviceType},
        future{tensor.future} {}

  Tensor(Tensor &&tensor) noexcept
      : data_{tensor.data_},
        layout{tensor.layout},
        dtype{tensor.dtype},
        deviceType{tensor.deviceType},
        future{std::move(tensor.future)} {
    tensor.data_ = nullptr;
    tensor.layout = {};
  }

  Tensor(const void *data, Layout layout, Dtype dtype, DeviceType deviceType,
         std::shared_ptr<Future> future = {})
      : data_{data},
        layout{layout},
        dtype{dtype},
        deviceType{deviceType},
        future{std::move(future)} {}

  void waitFutureIfValid() const {
    if (future != nullptr && future->valid()) {
      future->wait();
    }
  }

  void *data() { return const_cast<void *>(data_); }

  const void *data() const { return data_; }

 public:
  Layout layout;
  Dtype dtype;
  DeviceType deviceType;
  std::shared_ptr<Future> future;

 private:
  const void *data_;
};

template <template <int> class T, int N>
concept isTensor = std::is_same_v<std::decay_t<T<N>>, Tensor<N>>;

using Tensor1D = Tensor<1>;
using Tensor2D = Tensor<2>;  // (Row, Col)
using Tensor3D = Tensor<3>;  // (Batch, Sequence, Feature)
using Tensor4D = Tensor<4>;

template <int OutN, int InN>
  requires(InN > 1 && OutN < InN)
Tensor<OutN> flatten(const Tensor<InN> &tensor) {
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

template <int OutN, template <int> class T, int InN>
  requires isTensor<T, InN> && (InN > 1) && (OutN < InN)
auto flatten(const std::shared_ptr<T<InN>> &tensor) {
  return std::make_shared<Tensor<OutN>>(
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
}  // namespace dllm
