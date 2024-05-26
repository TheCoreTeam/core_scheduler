#pragma once

namespace dllm::util {
template <typename TA, typename TB>
constexpr __inline__ __attribute__((always_inline)) int ceil_div(TA a, TB b) {
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
