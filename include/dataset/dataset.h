#pragma once
#include <arrow/api.h>

#include <memory>

namespace dllm::dataset {
struct LlmDataset {
  std::shared_ptr<const arrow::Table> table;

  struct RowAccessor {
    RowAccessor() = delete;

    template <typename T>
    [[nodiscard]] T accessCol(std::size_t colIdx) const;

    ~RowAccessor();
  };

  [[nodiscard]] std::shared_ptr<const RowAccessor> accessRow(
      std::size_t rowIdx) const;
};
}  // namespace dllm::dataset
