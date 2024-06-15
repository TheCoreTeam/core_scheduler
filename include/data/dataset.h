#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace dllm::data {
struct Dataset {
  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

 protected:
  std::shared_ptr<Impl> impl_;
};

// This API is not stable
struct LlmDataset : Dataset {
  explicit LlmDataset(const std::string &directory);

  void fillBatch(const std::vector<std::int64_t *> &ptrs,
                 const std::vector<std::int64_t> &ld, std::int64_t startingRow,
                 std::int64_t batchSize) const;

  [[nodiscard]] std::int64_t attributeNum() const;

  [[nodiscard]] const std::vector<std::string> &attributeNames() const;

  [[nodiscard]] std::int64_t rows() const;

  [[nodiscard]] std::int64_t cols() const;
};
}  // namespace dllm::data
