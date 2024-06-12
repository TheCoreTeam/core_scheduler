#pragma once
#include "data/dataloader.h"

namespace dllm::data {
struct DataLoader::Impl {
  Impl(int64_t batchSize, int64_t numWorkers, bool shuffle, int64_t rank,
       int64_t worldSize);

  const int64_t batchSize;
  const int64_t numWorkers;
  const bool shuffle;
  const int64_t rank;
  const int64_t worldSize;

  [[nodiscard]] virtual int64_t iterationsPerEpoch() const = 0;

  virtual ~Impl() = default;
};
}  // namespace dllm::data
