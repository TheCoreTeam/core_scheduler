#pragma once
#include "data/dataloader.h"

namespace dllm::data {
struct DataLoader::Impl {
  Impl(int batchSize, int numWorkers, bool shuffle);

  const int batchSize;
  const int numWorkers;
  const bool shuffle;

  virtual ~Impl() = default;
};
}  // namespace dllm::data
