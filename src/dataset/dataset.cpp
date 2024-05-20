#include "dataset/dataset.h"

namespace dllm::dataset {
namespace {
struct RowAccessorImpl {
  const std::shared_ptr<const arrow::ListArray> row;
  const std::size_t rowOffset;

  template <typename T>
  T accessCol(std::size_t colIdx) const;
};
}  // namespace

std::shared_ptr<const LlmDataset::RowAccessor> LlmDataset::accessRow(
    const std::size_t rowIdx) const {
  const auto column = table->column(0);
  return std::reinterpret_pointer_cast<const RowAccessor>(
      std::make_shared<RowAccessorImpl>(
          std::static_pointer_cast<arrow::ListArray>(column->chunk(0)),
          rowIdx));
}

LlmDataset::RowAccessor::~RowAccessor() {
  const auto impl = reinterpret_cast<RowAccessorImpl *>(this);
  impl->~RowAccessorImpl();
}

template <typename T>
T LlmDataset::RowAccessor::accessCol(const std::size_t colIdx) const {
  const auto impl = reinterpret_cast<const RowAccessorImpl *>(this);
  return impl->accessCol<T>(colIdx);
}

template <typename T>
T RowAccessorImpl::accessCol(const std::size_t colIdx) const {
  const auto values =
      std::static_pointer_cast<arrow::NumericArray<T>>(row->values());
  return values->Value(rowOffset + colIdx);
}
}  // namespace dllm::dataset