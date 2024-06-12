#include "data/dataset.h"

#include <arrow/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>

#include <memory>

#include "data/dataset_impl.h"
#include "logger.h"

namespace dllm::data {
struct LlmDataset::RowAccessor::Impl {
  const std::shared_ptr<const arrow::ListArray> inputIdsRow;
  const std::int64_t inputIdsRowOffset;
  const std::shared_ptr<const arrow::ListArray> targetsRow;
  const std::int64_t targetsRowOffset;

  [[nodiscard]] Element accessCol(std::int64_t colIdx) const;

  [[nodiscard]] std::int64_t cols() const;
};

LlmDataset::Element LlmDataset::RowAccessor::Impl::accessCol(
    const std::int64_t colIdx) const {
  std::int64_t input_id;
  if (inputIdsRow->values()->type_id() == arrow::Type::INT64) {
    const auto inputsIdsValues =
        std::static_pointer_cast<arrow::Int64Array>(inputIdsRow->values());
    input_id = inputsIdsValues->Value(inputIdsRowOffset + colIdx);
  } else if (inputIdsRow->values()->type_id() == arrow::Type::INT32) {
    const auto inputsIdsValues =
        std::static_pointer_cast<arrow::Int32Array>(inputIdsRow->values());
    input_id = inputsIdsValues->Value(inputIdsRowOffset + colIdx);
  } else {
    throw;
  }

  std::int64_t target;
  if (targetsRow->values()->type_id() == arrow::Type::INT64) {
    const auto targetsValues =
        std::static_pointer_cast<arrow::Int64Array>(targetsRow->values());
    target = targetsValues->Value(targetsRowOffset + colIdx);
  } else if (targetsRow->values()->type_id() == arrow::Type::INT32) {
    const auto targetsValues =
        std::static_pointer_cast<arrow::Int32Array>(targetsRow->values());
    target = targetsValues->Value(targetsRowOffset + colIdx);
  } else {
    throw;
  }
  return {input_id, target};
}

std::int64_t LlmDataset::RowAccessor::Impl::cols() const {
  return inputIdsRow->length();
}

struct LlmDatasetImpl final : Dataset::Impl {
  std::shared_ptr<const arrow::Table> table;

  LlmDatasetImpl(std::shared_ptr<const arrow::Table> table)
      : table{std::move(table)} {}

  [[nodiscard]] std::int64_t rows() const { return table->column(0)->length(); }

  [[nodiscard]] LlmDataset::RowAccessor accessRow(
      const std::int64_t rowIdx) const {
    const auto inputIdsArray =
        std::static_pointer_cast<arrow::ListArray>(table->column(0)->chunk(0));
    const auto targetsArray =
        std::static_pointer_cast<arrow::ListArray>(table->column(1)->chunk(0));
    auto impl = std::make_unique<LlmDataset::RowAccessor::Impl>(
        LlmDataset::RowAccessor::Impl{
            std::static_pointer_cast<arrow::ListArray>(inputIdsArray),
            inputIdsArray->value_offset(rowIdx),
            std::static_pointer_cast<arrow::ListArray>(targetsArray),
            targetsArray->value_offset(rowIdx)});
    return LlmDataset::RowAccessor{std::move(impl)};
  }
};

const std::shared_ptr<Dataset::Impl> &Dataset::impl() const { return impl_; }

LlmDataset::LlmDataset(const std::vector<std::string> &path) {
  const auto filesystem = std::make_shared<arrow::fs::LocalFileSystem>();
  const auto format = std::make_shared<arrow::dataset::ParquetFileFormat>();

  auto result = arrow::dataset::FileSystemDatasetFactory::Make(filesystem, path,
                                                               format, {});
  DLLM_ASSERT_TRUE(result.ok(),
                   fmt::format("Failed to make dataset factory: {}",
                               result.status().ToString()));
  const auto factory = result.ValueOrDie();
  const auto dataset_result = factory->Finish();
  DLLM_ASSERT_TRUE(dataset_result.ok(),
                   fmt::format("Failed to finish dataset: {}",
                               dataset_result.status().ToString()));
  const auto &dataset = dataset_result.ValueOrDie();

  auto scanner_builder_result = dataset->NewScan();
  DLLM_ASSERT_TRUE(scanner_builder_result.ok(),
                   fmt::format("Failed to start a new scan: {}",
                               scanner_builder_result.status().ToString()));

  const auto scanner_builder = scanner_builder_result.ValueOrDie();

  const auto project_status = scanner_builder->Project({"input_ids", "labels"});
  DLLM_ASSERT_TRUE(project_status.ok(),
                   fmt::format("Failed to set projection columns: {}",
                               project_status.ToString()));

  const auto scanner_result = scanner_builder->Finish();
  DLLM_ASSERT_TRUE(scanner_result.ok(),
                   fmt::format("Failed to create scanner: {}",
                               scanner_result.status().ToString()));
  const auto &scanner = scanner_result.ValueOrDie();

  auto table_result = scanner->ToTable();
  DLLM_ASSERT_TRUE(
      table_result.ok(),
      fmt::format("Error reading table: {}", table_result.status().ToString()));
  auto table = table_result.ValueOrDie();
  impl_ = std::make_shared<LlmDatasetImpl>(std::move(table));
}

std::int64_t LlmDataset::RowAccessor::cols() const { return impl_->cols(); }

LlmDataset::RowAccessor::RowAccessor(std::unique_ptr<Impl> impl)
    : impl_{std::move(impl)} {}

LlmDataset::Element LlmDataset::RowAccessor::accessCol(
    const std::int64_t colIdx) const {
  return impl_->accessCol(colIdx);
}

LlmDataset::RowAccessor LlmDataset::accessRow(const std::int64_t rowIdx) const {
  return std::dynamic_pointer_cast<LlmDatasetImpl>(impl_)->accessRow(rowIdx);
}

std::int64_t LlmDataset::rows() const {
  return std::dynamic_pointer_cast<LlmDatasetImpl>(impl_)->rows();
}

std::int64_t LlmDataset::cols() const {
  const auto row = accessRow(0);
  return row.cols();
}
}  // namespace dllm::data
