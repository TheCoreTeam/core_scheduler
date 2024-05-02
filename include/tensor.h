#pragma once
#include <mpi.h>
#include <nccl.h>

#include <cute/layout.hpp>
#include <future>
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

  void *data;
  Layout layout;
  Dtype dtype;
  DeviceType deviceType;
  std::shared_ptr<Future> future;

  void waitFutureIfValid() const {
    if (future != nullptr && future->valid()) {
      future->wait();
    }
  }
};

template <int N>
struct DistributedTensor : public Tensor<N> {
 private:
  using Base = Tensor<N>;
  using Coord = typename repeat_type<int, N, cute::Coord>::type;

 public:
  Coord localCoord;
  Base::Layout localLayout;
};

using Tensor1D = Tensor<1>;
using Tensor2D = Tensor<2>;  // (Row, Col)
using Tensor3D = Tensor<3>;  // (Batch, Sequence, Feature)
using Tensor4D = Tensor<4>;

using DistributedTensor1D = DistributedTensor<1>;
using DistributedTensor2D = DistributedTensor<2>;  // (Row, Col)
using DistributedTensor3D = DistributedTensor<3>;  // (Batch, Sequence, Feature)
using DistributedTensor4D = DistributedTensor<4>;
}  // namespace dllm
