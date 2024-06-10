#pragma once
#include <memory>

#include "dataset/dataset.h"
#include "tensor.h"

namespace dllm::dataset {
// This class is only a proxy class
struct LlmDataLoader {
  // fill will set the future, so you do not have to sync with x and y's future
  // handles explicitly
  void fill(const Tensor &x, const Tensor &y) const;

  LlmDataLoader(const std::shared_ptr<const LlmDataset> &dataset, int localRank,
                int batch_size, int num_workers, bool shuffle,
                const std::vector<int> &bindingMap = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dllm::dataset
