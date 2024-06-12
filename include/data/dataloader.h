#pragma once
#include <memory>

#include "data/dataset.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::data {
struct DataLoader {
  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

 protected:
  std::shared_ptr<Impl> impl_;
};

// This API is not stable
struct LlmDataLoader : DataLoader {
  void load(const Scheduler &scheduler, Tensor &x, Tensor &y) const;

  LlmDataLoader(const LlmDataset &dataset, int batchSize, int numWorkers,
                bool shuffle);
};
}  // namespace dllm::data
