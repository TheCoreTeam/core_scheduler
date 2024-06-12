#pragma once
#include <memory>

namespace dllm {
namespace dataset {
struct DataLoader;
}

struct Scheduler {
  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

  [[nodiscard]] int64_t deviceRank() const;

 protected:
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm
