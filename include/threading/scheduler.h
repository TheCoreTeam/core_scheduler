#pragma once
#include <memory>

namespace dllm {
struct Scheduler {
  struct Impl;

 protected:
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm
