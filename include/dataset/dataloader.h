#pragma once
#include <memory>

#include "dataset/dataset.h"

namespace dllm::dataset {
struct LlmDataLoader {
  explicit LlmDataLoader(const std::shared_ptr<const LlmDataset> &dataset,
                         std::size_t batch_size, std::size_t num_worker,
                         bool shuffle);


};
}  // namespace dllm::dataset
