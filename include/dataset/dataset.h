#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace dllm::dataset {
struct LlmDataset {
  LlmDataset() = delete;

  static std::shared_ptr<const LlmDataset> create(
      const std::vector<std::string> &path);

  struct Element {
    std::int64_t input_id;
    std::int64_t target;
  };

  struct RowAccessor {
    RowAccessor() = delete;

    [[nodiscard]] Element accessCol(std::int64_t colIdx) const;

    std::int64_t cols() const;

    ~RowAccessor();
  };

  [[nodiscard]] std::shared_ptr<const RowAccessor> accessRow(
      std::int64_t rowIdx) const;

  std::int64_t rows() const;

  std::int64_t cols() const;
};
}  // namespace dllm::dataset
