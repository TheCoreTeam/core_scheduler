#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::util {
// template <OpType op_, TensorBackend backend>
// struct MarkedTensor {
//   constexpr static auto op = op_;
//   const std::shared_ptr<Tensor<backend>> &tensor;
//
//   auto extract_value() const { return tensor; }
//
//   auto extract_future() const {
//     return TensorFriend::accessImpl(*tensor).future();
//   }
//
//   void refresh_future(const TaskFuture &future) const {
//     TensorFriend::accessImpl(*tensor).future().wFuture = future;
//   }
// };
//
// template <TensorBackend backend>
// struct MarkedTensor<R, backend> {
//   constexpr static auto op = R;
//   const std::shared_ptr<Tensor<backend>> &tensor;
//
//   auto extract_value() const { return tensor; }
//
//   auto extract_future() const {
//     return TensorFriend::accessImpl(*tensor).future()->wFuture;
//   }
//
//   void refresh_future(const TaskFuture &future) const {
//     TensorFriend::accessImpl(*tensor).future().rFuture = future;
//   }
// };
//
// template <OpType op, TensorBackend backend>
// auto markTensor(const std::shared_ptr<Tensor<backend>> &tensor) {
//   return MarkedTensor<op, backend>{tensor};
// }
//
// template <OpType op_, TensorBackend backend>
// struct MarkedConstTensor {
//   constexpr static auto op = op_;
//   const std::shared_ptr<const Tensor<backend>> &tensor;
//
//   auto extract_value() const { return tensor; }
//
//   auto extract_future() const {
//     return TensorFriend::accessImpl(*tensor).future();
//   }
//
//   void refresh_future(const TaskFuture &future) const {
//     TensorFriend::accessImpl(*tensor).future().wFuture = future;
//   }
// };
//
// template <TensorBackend backend>
// struct MarkedConstTensor<R, backend> {
//   constexpr static auto op = R;
//   const std::shared_ptr<const Tensor<backend>> &tensor;
//
//   auto extract_value() const { return tensor; }
//
//   auto extract_future() const {
//     return TensorFriend::accessImpl(*tensor).future().wFuture;
//   }
//
//   void refresh_future(const TaskFuture &future) const {
//     TensorFriend::accessImpl(*tensor).future().rFuture = future;
//   }
// };
//
// template <OpType op, TensorBackend backend>
// auto markTensor(const std::shared_ptr<const Tensor<backend>> &tensor) {
//   return MarkedConstTensor<op, backend>{tensor};
// }
//
// template <typename T>
// struct MarkedTensorHelper {
//   constexpr static bool is_marked = false;
//
//   static auto extract_value(const T &t) { return t; }
// };
//
// template <OpType op, TensorBackend backend>
// struct MarkedTensorHelper<MarkedTensor<op, backend>> {
//   constexpr static bool is_marked = true;
//
//   static auto extract_value(const MarkedTensor<op, backend> &t) {
//     return t.extract_value();
//   }
// };
//
// template <OpType op, TensorBackend backend>
// struct MarkedTensorHelper<MarkedConstTensor<op, backend>> {
//   constexpr static bool is_marked = true;
//
//   static auto extract_value(const MarkedConstTensor<op, backend> &t) {
//     return t.extract_value();
//   }
// };
//
// template <typename Tuple, size_t... I>
// auto filter_tuple_impl(Tuple &&t, std::index_sequence<I...>) {
//   auto maybe_add = [](auto &&x) {
//     if constexpr (MarkedTensorHelper<std::decay_t<decltype(x)>>::is_marked)
//       return std::make_tuple(std::forward<decltype(x)>(x).extract_future());
//     else
//       return std::tuple<>{};
//   };
//
//   return std::tuple_cat(maybe_add(std::get<I>(std::forward<Tuple>(t)))...);
// }
//
// template <std::size_t I = 0, typename... Tp>
// std::enable_if_t<I == sizeof...(Tp), void> wait_future(
//     const std::tuple<Tp...> &t) {}
//
// template <std::size_t I = 0, typename... Tp>
//     std::enable_if_t <
//     I<sizeof...(Tp), void> wait_future(const std::tuple<Tp...> &t) {
//   std::get<I>(t).wait();
//   wait_future<I + 1, Tp...>(t);
// }
//
// template <std::size_t I = 0, typename... Tp>
// std::enable_if_t<I == sizeof...(Tp), void> reset_future(std::tuple<Tp...> &t)
// {}
//
// template <std::size_t I = 0, typename... Tp>
//     inline std::enable_if_t <
//     I<sizeof...(Tp), void> reset_future(std::tuple<Tp...> &t) {
//   std::get<I>(t) = {};
//   reset_future<I + 1, Tp...>(t);
// }
//
// template <typename T>
// struct is_smart_pointer : std::false_type {};
//
// template <typename T>
// struct is_smart_pointer<std::shared_ptr<T>> : std::true_type {};
//
// template <typename T>
// struct is_smart_pointer<std::unique_ptr<T>> : std::true_type {};
//
// template <std::size_t I = 0, typename... Tp>
// std::enable_if_t<I == sizeof...(Tp), void> reset_pointers(
//     std::tuple<Tp...> &t) {}
//
// template <std::size_t I = 0, typename... Tp>
//     std::enable_if_t <
//     I<sizeof...(Tp), void> reset_pointers(std::tuple<Tp...> &t) {
//   if constexpr (is_smart_pointer<
//                     std::tuple_element_t<I, std::tuple<Tp...>>>::value) {
//     std::get<I>(t).reset();
//   }
//   reset_pointers<I + 1, Tp...>(t);
// }
//
// template <std::size_t I = 0, typename... Tp>
// std::enable_if_t<I == sizeof...(Tp), void> refresh_future(
//     const std::tuple<Tp...> &t, const TaskFuture &future) {}
//
// template <std::size_t I = 0, typename... Tp>
//     std::enable_if_t <
//     I<sizeof...(Tp), void> refresh_future(const std::tuple<Tp...> &t,
//                                           const TaskFuture &future) {
//   if constexpr (MarkedTensorHelper<
//                     std::tuple_element_t<I, std::tuple<Tp...>>>::is_marked) {
//     std::get<I>(t).refresh_future(future);
//   }
//   refresh_future<I + 1, Tp...>(t, future);
// }
//
// template <typename... Args>
// auto extract_future_from_args(const Args &...args) {
//   return filter_tuple_impl(std::make_tuple(std::ref(args)...),
//                            std::index_sequence_for<Args...>{});
// }
//
// template <typename... Args>
// auto extract_value_from_args(const Args &...args) {
//   return std::make_tuple(MarkedTensorHelper<Args>::extract_value(args)...);
// }

constexpr __inline__ __attribute__((always_inline)) int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

constexpr __inline__ __attribute__((always_inline)) long ceilDiv(long a,
                                                                 long b) {
  return (a + b - 1) / b;
}

template <typename FutureType>
struct FutureGuard {
  FutureType &future;
  explicit FutureGuard(FutureType &future) : future{future} {
    if (future.valid()) {
      future.wait();
    }
  }

  ~FutureGuard() { future = {}; }

  void reset() const { future = {}; }
};
}  // namespace dllm::util
