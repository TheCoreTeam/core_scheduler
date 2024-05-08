#pragma once
#include <cute/layout.hpp>
#include <future>
#include <memory>
#include <tuple>
#include <vector>

#include "device.h"
#include "dtype.h"
#include "logger.h"
#include "threading/task_compute.h"

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

using TensorIndexType = long;

template <int N_>
struct Tensor {
  constexpr static int N = N_;
  static_assert(N >= 1);
  using Shape = typename repeat_type<TensorIndexType, N, cute::Shape>::type;
  using Stride =
      typename repeat_last_1_type<TensorIndexType, N - 1, cute::Stride>::type;
  using Layout = cute::Layout<Shape, Stride>;

  Tensor(const Tensor &tensor)
      : data_{tensor.data_},
        layout{tensor.layout},
        dtype{tensor.dtype},
        deviceType{tensor.deviceType},
        future{tensor.future} {}

  Tensor(Tensor &&tensor) noexcept
      : data_{std::move(tensor.data_)},
        layout{tensor.layout},
        dtype{tensor.dtype},
        deviceType{tensor.deviceType},
        future{std::move(tensor.future)} {
    tensor.layout = {};
  }

  Tensor(const void *data, Layout layout, Dtype dtype, DeviceType deviceType,
         std::shared_ptr<FutureCompute> future = {})
      : data_{std::shared_ptr<const void>{data, [](const void *) {}}},
        layout{layout},
        dtype{dtype},
        deviceType{deviceType},
        future{std::move(future)} {}

  template <template <typename T> class SmartPointer, typename T>
  Tensor(SmartPointer<T> &&data, Layout layout, Dtype dtype,
         DeviceType deviceType, std::shared_ptr<FutureCompute> future = {})
      : data_{std::forward<SmartPointer<T>>(data)},
        layout{layout},
        dtype{dtype},
        deviceType{deviceType},
        future{std::move(future)} {}

  void *data() { return const_cast<void *>(data_.get()); }

  const void *data() const { return data_.get(); }

  template <template <typename T> class SmartPointer, typename T>
  void resetData(SmartPointer<T> &&data) {
    data_ = std::forward<SmartPointer<T>>(data);
  }

#ifdef DLLM_BUILD_FLASH_ATTENTION
  // following functions are internal use to align with pytorch api
  // NEVER use them alone!
  template <std::size_t... I>
  __inline__ __attribute__((always_inline)) auto sizes_impl(
      const Layout &layout, std::index_sequence<I...>) const {
    return std::array<TensorIndexType, N>{cute::shape<I>(layout)...};
  }

  constexpr auto sizes() const {
    return sizes_impl(layout, std::make_index_sequence<N>{});
  }

  template <int k>
  constexpr auto size() const {
    if constexpr (k >= 0) {
      return cute::shape<k>(layout);
    } else {
      return cute::shape<Layout::rank + k>(layout);
    }
  }

  template <int k>
  constexpr auto stride() const {
    if constexpr (k >= 0) {
      return cute::stride<k>(layout);
    } else {
      return cute::stride<Layout::rank + k>(layout);
    }
  }

  auto numel() const { return cute::size(layout); }

  auto scalar_type() const { return dtype; }

  template <typename T = void>
  auto data_ptr() {
    return reinterpret_cast<T *>(data());
  };

  template <typename T = const void>
  auto data_ptr() const {
    return reinterpret_cast<T *>(data());
  };

  auto device() const {
    struct Device {
      DeviceType deviceType;
      auto type() const { return deviceType; }
    } device{.deviceType = deviceType};
    return device;
  }

  bool is_cuda() const { return deviceType == CUDA; }

  constexpr TensorIndexType dim() const { return Layout::rank; }

  template <DeviceType deviceType, typename Layout, typename... Args>
  static std::shared_ptr<Tensor> empty(const Layout &layout, Dtype dtype,
                                       Args &&...args) {
    if constexpr (deviceType == CUDA) {
      constexpr auto argsIsContextComputPointer =
          std::is_same_v<std::tuple<std::remove_const_t<
                             std::remove_pointer_t<std::decay_t<Args>>>...>,
                         std::tuple<ContextCompute>>;
      static_assert(argsIsContextComputPointer);
      const auto size = cute::cosize(layout);
      if constexpr (argsIsContextComputPointer) {
        const ContextCompute *context = std::get<0>(std::tuple<Args>{args}...);
        std::shared_ptr<void> data{
            [&] {
              void *ptr;
              CHECK_CUDART(cudaMallocFromPoolAsync(&ptr, toByte(dtype) * size,
                                                   context->memPool,
                                                   context->cudaStream));
              CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
              return ptr;
            }(),
            [stream = context->cudaStream](void *ptr) {
              CHECK_CUDART(cudaFreeAsync(ptr, stream));
            }};
        return std::make_shared<Tensor>(std::move(data), layout, dtype, CUDA);
      } else {
      }
    } else {
    }
  }
  // above functions are internal use to align with pytorch api
  // NEVER use them alone!
#endif

 public:
  Layout layout;
  Dtype dtype;
  DeviceType deviceType;
  std::shared_ptr<FutureCompute> future;

 private:
  std::shared_ptr<const void> data_;
};

template <template <int> class T, int N>
constexpr bool isTensor =
    std::is_same_v<std::remove_const_t<std::decay_t<T<N>>>, Tensor<N>>;

using Tensor1D = Tensor<1>;
using Tensor2D = Tensor<2>;  // (Row, Col)
using Tensor3D = Tensor<3>;  // (Batch, Sequence, Feature)
using Tensor4D = Tensor<4>;
}  // namespace dllm
