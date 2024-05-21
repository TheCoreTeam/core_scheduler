#include "dataset/dataset.h"

#include <arrow/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>

#include "logger.h"

namespace dllm::dataset {
namespace {
struct RowAccessorImpl {
  const std::shared_ptr<const arrow::ListArray> inputIdsRow;
  const std::int64_t inputIdsRowOffset;
  const std::shared_ptr<const arrow::ListArray> targetsRow;
  const std::int64_t targetsRowOffset;

  [[nodiscard]] LlmDataset::Element accessCol(const std::int64_t colIdx) const {
    // TODO(Jie): maybe use template for the Accessor Class
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

  [[nodiscard]] std::int64_t cols() const { return inputIdsRow->length(); }
};

struct LlmDatasetImpl {
  std::shared_ptr<const arrow::Table> table;

  [[nodiscard]] std::int64_t rows() const { return table->column(0)->length(); }

  [[nodiscard]] std::shared_ptr<const LlmDataset::RowAccessor> accessRow(
      const std::int64_t rowIdx) const {
    const auto inputIdsArray =
        std::static_pointer_cast<arrow::ListArray>(table->column(0)->chunk(0));
    const auto targetsArray =
        std::static_pointer_cast<arrow::ListArray>(table->column(1)->chunk(0));
    return std::reinterpret_pointer_cast<const LlmDataset::RowAccessor>(
        std::make_shared<RowAccessorImpl>(
            std::static_pointer_cast<arrow::ListArray>(inputIdsArray),
            inputIdsArray->value_offset(rowIdx),
            std::static_pointer_cast<arrow::ListArray>(targetsArray),
            targetsArray->value_offset(rowIdx)));
  }
};
}  // namespace

std::shared_ptr<const LlmDataset> LlmDataset::create(
    const std::vector<std::string> &path) {
  const auto filesystem = std::make_shared<arrow::fs::LocalFileSystem>();
  const auto format = std::make_shared<arrow::dataset::ParquetFileFormat>();

  auto result = arrow::dataset::FileSystemDatasetFactory::Make(filesystem, path,
                                                               format, {});
  if (!result.ok()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Failed to make dataset factory: {}",
                           result.status().ToString());
    return {};
  }
  const auto factory = result.ValueOrDie();
  const auto dataset_result = factory->Finish();
  if (!dataset_result.ok()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Failed to finish dataset: {}",
                           dataset_result.status().ToString());
    return {};
  }
  const auto &dataset = dataset_result.ValueOrDie();

  auto scanner_builder_result = dataset->NewScan();
  if (!scanner_builder_result.ok()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Failed to start a new scan: {}",
                           scanner_builder_result.status().ToString());
    return {};
  }
  const auto scanner_builder = scanner_builder_result.ValueOrDie();

  const auto project_status = scanner_builder->Project({"input_ids", "labels"});
  if (!project_status.ok()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Failed to set projection columns: {}",
                           project_status.ToString());
    return {};
  }

  const auto scanner_result = scanner_builder->Finish();
  if (!scanner_result.ok()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Failed to create scanner: {}",
                           scanner_result.status().ToString());
    return {};
  }
  const auto &scanner = scanner_result.ValueOrDie();

  auto table_result = scanner->ToTable();
  if (!table_result.ok()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Error reading table: {}",
                           table_result.status().ToString());
    return {};
  }
  auto table = table_result.ValueOrDie();
  return std::reinterpret_pointer_cast<const LlmDataset>(
      std::make_shared<LlmDatasetImpl>(table));
}

LlmDataset::RowAccessor::~RowAccessor() {
  const auto impl = reinterpret_cast<RowAccessorImpl *>(this);
  impl->~RowAccessorImpl();
}

std::int64_t LlmDataset::RowAccessor::cols() const {
  const auto impl = reinterpret_cast<const RowAccessorImpl *>(this);
  return impl->cols();
}

LlmDataset::Element LlmDataset::RowAccessor::accessCol(
    const std::int64_t colIdx) const {
  const auto impl = reinterpret_cast<const RowAccessorImpl *>(this);
  return impl->accessCol(colIdx);
}

std::shared_ptr<const LlmDataset::RowAccessor> LlmDataset::accessRow(
    const std::int64_t rowIdx) const {
  const auto impl = reinterpret_cast<const LlmDatasetImpl *>(this);
  return impl->accessRow(rowIdx);
}

std::int64_t LlmDataset::rows() const {
  const auto impl = reinterpret_cast<const LlmDatasetImpl *>(this);
  return impl->rows();
}

std::int64_t LlmDataset::cols() const {
  const auto row = accessRow(0);
  return row->cols();
}
}  // namespace dllm::dataset
