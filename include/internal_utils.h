#pragma once
#ifdef DLLM_ENABLE_INTERNAL_BUILD
// This is an internal file, never use it unless you know what you are doing
namespace dllm::utils {
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
      try {
        future.get();
      } catch (const std::exception &) {
        std::rethrow_exception(std::current_exception());
      }
    }
  }

  ~FutureGuard() { future = {}; }

  void reset() const { future = {}; }
};
}  // namespace dllm::util
#else
#error "You should not include this file in your program!"
#endif
