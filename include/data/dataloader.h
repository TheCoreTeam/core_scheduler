#pragma once
#include <memory>

#include "data/dataset.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::data {
struct DataLoader {
  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

  [[nodiscard]] int64_t iterationsPerEpoch() const;

 protected:
  std::shared_ptr<Impl> impl_;
};

// This API is not stable
struct LlmDataLoader : DataLoader {
  std::unordered_map<std::string, Tensor> load(
      const Scheduler &scheduler) const;

  LlmDataLoader(const LlmDataset &dataset, int64_t batchSize,
                int64_t numWorkers, bool shuffle, int64_t rank,
                int64_t worldSize);
};
}  // namespace dllm::data
