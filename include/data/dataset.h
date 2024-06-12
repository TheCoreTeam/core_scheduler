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
  explicit LlmDataset(const std::vector<std::string> &path);

  struct Element {
    std::int64_t input_id;
    std::int64_t target;
  };

  struct RowAccessor {
    struct Impl;

    explicit RowAccessor(std::unique_ptr<Impl> impl);

    [[nodiscard]] Element accessCol(std::int64_t colIdx) const;

    [[nodiscard]] std::int64_t cols() const;

   private:
    std::unique_ptr<Impl> impl_;
  };

  [[nodiscard]] RowAccessor accessRow(std::int64_t rowIdx) const;

  [[nodiscard]] std::int64_t rows() const;

  [[nodiscard]] std::int64_t cols() const;
};
}  // namespace dllm::data
