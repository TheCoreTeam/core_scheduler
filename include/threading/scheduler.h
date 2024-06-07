#pragma once
#include <memory>

namespace dllm {
struct Scheduler {
  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

 protected:
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm
