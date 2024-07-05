/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "data/dataset.h"

#include <arrow/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>
#include <parquet/file_reader.h>

#include <filesystem>
#include <memory>

#include "data/dataset_impl.h"
#include "logger.h"

namespace cs::data {
const std::shared_ptr<Dataset::Impl> &Dataset::impl() const { return impl_; }

LlmDatasetImpl::LlmDatasetImpl(std::vector<std::string> files,
                               std::vector<int64_t> rowOffset)
    : files_{std::move(files)}, fileOffsets_{std::move(rowOffset)} {}

std::int64_t LlmDatasetImpl::size() const {
  return fileOffsets_[fileOffsets_.size() - 1];
}

const std::vector<std::string> &LlmDatasetImpl::files() const { return files_; }

const std::vector<int64_t> &LlmDatasetImpl::fileOffsets() const {
  return fileOffsets_;
}

namespace {
namespace fs = std::filesystem;

std::vector<std::string> get_arrow_files(const std::string &directory) {
  std::vector<std::string> files;

  // 确保提供的路径是一个目录
  CS_ASSERT_TRUE(fs::is_directory(directory),
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

int64_t getParquetRowCount(const std::string &file_path) {
  // Open the Parquet file
  const auto infile = arrow::io::ReadableFile::Open(file_path).ValueOrDie();

  // Create a Parquet file reader
  const auto parquet_reader = parquet::ParquetFileReader::Open(infile);

  // Get the file metadata
  const auto file_metadata = parquet_reader->metadata();

  // Return the total number of rows in the file
  return file_metadata->num_rows();
}
}  // namespace

LlmDataset::LlmDataset(const std::string &directory) {
  const auto filesystem = std::make_shared<arrow::fs::LocalFileSystem>();
  const auto format = std::make_shared<arrow::dataset::ParquetFileFormat>();
  auto parquetFiles = get_arrow_files(directory);

  std::vector<int64_t> rowOffset;
  rowOffset.reserve(parquetFiles.size() + 1);
  rowOffset.push_back(0);
  {
    int64_t total_rows = 0;

    for (const auto &path : parquetFiles) {
      total_rows += getParquetRowCount(path);
      rowOffset.push_back(total_rows);
    }
  }

  impl_ = std::make_shared<LlmDatasetImpl>(std::move(parquetFiles),
                                           std::move(rowOffset));
}

std::int64_t LlmDataset::size() const {
  return std::dynamic_pointer_cast<LlmDatasetImpl>(impl_)->size();
}

const std::vector<std::string> &LlmDataset::files() const {
  return std::dynamic_pointer_cast<LlmDatasetImpl>(impl_)->files();
}

const std::vector<int64_t> &LlmDataset::fileOffsets() const {
  return std::dynamic_pointer_cast<LlmDatasetImpl>(impl_)->fileOffsets();
}
}  // namespace cs::data
