#pragma once
#include <atomic>
#include <cstdint>

namespace dllm::random {
struct RandomState {
  uint64_t seed;
  std::atomic<uint64_t> offset;
};
}  // namespace dllm::random