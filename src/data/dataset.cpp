#include "data/dataset.h"

#include <arrow/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>

#include <filesystem>
#include <memory>

#include "data/dataset_impl.h"
#include "logger.h"

namespace dllm::data {
struct LlmDatasetImpl final : Dataset::Impl {
  std::shared_ptr<const arrow::Table> table;
  const std::int64_t attributeNum;
  const std::vector<std::string> attributeNames;
  const std::vector<int64_t> rowOffset_;

  LlmDatasetImpl(std::shared_ptr<const arrow::Table> table,
                 const std::int64_t attributeNum,
                 std::vector<std::string> attributeNames,
                 std::vector<int64_t> rowOffset)
      : table{std::move(table)},
        attributeNum{attributeNum},
        attributeNames{std::move(attributeNames)},
        rowOffset_{std::move(rowOffset)} {}

  [[nodiscard]] std::int64_t rows() const {
    return rowOffset_[rowOffset_.size() - 1];
  }
};

const std::shared_ptr<Dataset::Impl> &Dataset::impl() const { return impl_; }

namespace {
namespace fs = std::filesystem;

std::vector<std::string> get_arrow_files(const std::string &directory) {
  std::vector<std::string> files;

  // 确保提供的路径是一个目录
  DLLM_ASSERT_TRUE(fs::is_directory(directory),
                   "Provided path is not a directory.");

  // 遍历目录中的所有项
  for (const auto &entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file()) {
      if (const auto &path = entry.path(); path.extension() == ".parquet") {
        files.push_back(fs::absolute(path));  // 获取绝对路径
      }
    }
  }

  // 对路径进行排序，保证有序
  std::ranges::sort(files);

  return files;
}
}  // namespace

LlmDataset::LlmDataset(const std::string &directory) {
  const auto filesystem = std::make_shared<arrow::fs::LocalFileSystem>();
  const auto format = std::make_shared<arrow::dataset::ParquetFileFormat>();

  auto result = arrow::dataset::FileSystemDatasetFactory::Make(
      filesystem, get_arrow_files(directory), format, {});
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
  std::vector<std::string> attributeNames;
  attributeNames.reserve(table->num_columns());
  for (int i = 0; i < table->num_columns(); ++i) {
    attributeNames.push_back(table->schema()->field(i)->name());
  }
  std::vector<int64_t> rowOffset;
  rowOffset.reserve(table->column(0)->num_chunks() + 1);
  int64_t rows = 0;
  rowOffset.push_back(0);
  for (int i = 0; i < table->column(0)->num_chunks(); ++i) {
    rows += table->column(0)->chunk(i)->length();
    rowOffset.push_back(rows);
  }
  impl_ = std::make_shared<LlmDatasetImpl>(
      std::move(table), attributeNames.size(), std::move(attributeNames),
      std::move(rowOffset));
}

void LlmDataset::fillBatch(const std::vector<std::int64_t *> &ptrs,
                           const std::vector<std::int64_t> &ld,
                           const std::int64_t startingRow,
                           const std::int64_t batchSize) const {
  const auto impl = std::dynamic_pointer_cast<LlmDatasetImpl>(impl_);
  DLLM_ASSERT_TRUE(impl->table->num_columns() == static_cast<int>(ptrs.size()),
                   "wrong ptrs size");
  int startChunkIdx = -1;
  for (int i = 1; i < static_cast<int>(impl->rowOffset_.size()); ++i) {
    if (impl->rowOffset_[i] >= startingRow) {
      startChunkIdx = i - 1;
      break;
    }
  }
  int endChunkIdx = -1;
  for (int i = startChunkIdx; i < static_cast<int>(impl->rowOffset_.size());
       ++i) {
    if (impl->rowOffset_[i] >= startingRow + batchSize) {
      endChunkIdx = i - 1;
      break;
    }
  }
  DLLM_ASSERT_TRUE(startChunkIdx != -1,
                   "Data loader error, one reason is batchsize is larger than "
                   "the dataset size");
  DLLM_ASSERT_TRUE(endChunkIdx != -1,
                   "Data loader error, one reason is batchsize is larger than "
                   "the dataset size");

  for (int i = 0; i < static_cast<int>(ptrs.size()); ++i) {
    auto data = std::static_pointer_cast<arrow::ListArray>(
        impl->table->column(i)->chunk(0));

    const int64_t cols = data->value_offset(1) - data->value_offset(0);

    if (data->value_type()->id() == arrow::Type::INT64) {
      int64_t rowOffset = 0;
      for (int chunkIdx = startChunkIdx; chunkIdx < endChunkIdx + 1;
           ++chunkIdx) {
        auto data = std::static_pointer_cast<arrow::ListArray>(
            impl->table->column(i)->chunk(chunkIdx));
        for (int64_t row =
                 std::max<int64_t>(startingRow - impl->rowOffset_[chunkIdx], 0);
             row <
             std::min(
                 startingRow + batchSize - impl->rowOffset_[chunkIdx],
                 impl->rowOffset_[chunkIdx + 1] - impl->rowOffset_[chunkIdx]);
             ++row) {
          const auto offset = data->value_offset(row);
          for (int64_t col = 0; col < cols; ++col) {
            ptrs[i][rowOffset * ld[i] + col] =
                std::static_pointer_cast<arrow::Int64Array>(data->values())
                    ->Value(offset + col);
          }
          ++rowOffset;
        }
      }
      assert(rowOffset == batchSize);
    } else if (data->value_type()->id() == arrow::Type::INT32) {
      int64_t rowOffset = 0;
      for (int chunkIdx = startChunkIdx; chunkIdx < endChunkIdx + 1;
           ++chunkIdx) {
        auto data = std::static_pointer_cast<arrow::ListArray>(
            impl->table->column(i)->chunk(chunkIdx));
        for (int64_t row =
                 std::max<int64_t>(startingRow - impl->rowOffset_[chunkIdx], 0);
             row <
             std::min(
                 startingRow + batchSize - impl->rowOffset_[chunkIdx],
                 impl->rowOffset_[chunkIdx + 1] - impl->rowOffset_[chunkIdx]);
             ++row) {
          const auto offset = data->value_offset(row);
          for (int64_t col = 0; col < cols; ++col) {
            ptrs[i][rowOffset * ld[i] + col] =
                std::static_pointer_cast<arrow::Int32Array>(data->values())
                    ->Value(offset + col);
          }
          ++rowOffset;
        }
      }
      assert(rowOffset == batchSize);
    }
  }
}

std::int64_t LlmDataset::attributeNum() const {
  return std::dynamic_pointer_cast<LlmDatasetImpl>(impl_)->attributeNum;
}

const std::vector<std::string> &LlmDataset::attributeNames() const {
  return std::dynamic_pointer_cast<LlmDatasetImpl>(impl_)->attributeNames;
}

std::int64_t LlmDataset::rows() const {
  return std::dynamic_pointer_cast<LlmDatasetImpl>(impl_)->rows();
}

std::int64_t LlmDataset::cols() const {
  const auto impl = std::dynamic_pointer_cast<LlmDatasetImpl>(impl_);
  const auto data = std::static_pointer_cast<arrow::ListArray>(
      impl->table->column(0)->chunk(0));
  return data->value_offset(1) - data->value_offset(0);
}
}  // namespace dllm::data
