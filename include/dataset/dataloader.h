#pragma once
#include <memory>

#include "dataset/dataset.h"
#include "tensor.h"
#include "threading/task_cudart.h"

namespace dllm::dataset {
// This class is only a proxy class
struct LlmDataLoader {
  LlmDataLoader() = delete;

  ~LlmDataLoader();

  // fill will set the future, so you do not have to sync with x and y's future
  // handles explicitly
  void fill(const std::shared_ptr<Tensor2D> &x,
            const std::shared_ptr<Tensor2D> &y) const;

  static std::shared_ptr<const LlmDataLoader> create(
      const std::shared_ptr<const LlmDataset> &dataset, int localRank,
      int batch_size, int num_workers, bool shuffle,
      const std::vector<int> &bindingMap = {});
};
}  // namespace dllm::dataset
