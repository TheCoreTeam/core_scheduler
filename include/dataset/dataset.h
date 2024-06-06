#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace dllm::dataset {
struct LlmDataset {
  LlmDataset(const std::vector<std::string> &path);

  struct Element {
    std::int64_t input_id;
    std::int64_t target;
  };

  struct RowAccessor {
    struct Impl;
    RowAccessor(std::unique_ptr<Impl> impl) : impl_{std::move(impl)} {}
    [[nodiscard]] Element accessCol(std::int64_t colIdx) const;

    [[nodiscard]] std::int64_t cols() const;

   private:
    std::unique_ptr<Impl> impl_;
  };

  [[nodiscard]] RowAccessor accessRow(std::int64_t rowIdx) const;

  [[nodiscard]] std::int64_t rows() const;

  [[nodiscard]] std::int64_t cols() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dllm::dataset
