#pragma once
#include <atomic>

namespace dllm::random {
struct RandomState {
  uint64_t seed;
  std::atomic<uint64_t> offset;
};

RandomState &getRandomState();
}  // namespace dllm::random
