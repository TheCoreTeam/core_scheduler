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

#include "data/dataloader.h"

#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <arrow/array/array_nested.h>
#include <arrow/array/array_primitive.h>
#include <arrow/dataset/discovery.h>
#include <arrow/dataset/file_parquet.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <barrier>
#include <cstdint>
#include <span>

#include "data/dataloader_impl.h"
#include "data/dataset.h"
#include "data/dataset_impl.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace arrow {
class ListArray;
}

namespace cs::data {
inline DataLoader::Impl::Impl(const int64_t batchSize, const int64_t numWorkers,
                              const bool shuffle, const int64_t rank,
                              const int64_t worldSize)
    : batchSize{batchSize},
      num_workers{numWorkers},
      shuffle{shuffle},
      rank{rank},
      world_size{worldSize} {}

struct HostBuffer {
  void *ptr{nullptr};
  std::size_t size{0};
  IntArray sizes{};
  TensorOptions options;
  std::size_t validDataSize{0};

  ~HostBuffer();
};

struct LlmDataLoaderImpl final : DataLoader::Impl {
  LlmDataLoaderImpl(const LlmDataset &dataset, const int64_t batch_size,
                    const int64_t num_workers, const bool shuffle,
                    const int64_t rank, const int64_t worldSize)
      : Impl{batch_size, num_workers, shuffle, rank, worldSize},
        dataset_{dataset} {}

  ~LlmDataLoaderImpl() override;

  [[nodiscard]] int64_t iterations_per_epoch() const override;

  const LlmDataset dataset_;
  std::vector<std::shared_ptr<std::barrier<>>> startBarrier_{};
  std::vector<Event> events_{};
  std::vector<std::shared_ptr<HostBuffer[]>> buffers_{};
  std::vector<std::shared_ptr<std::barrier<>>> endBarrier_{};
  std::vector<std::jthread> threadVector_{};
  std::shared_ptr<std::atomic<bool>> shutDown_{nullptr};
  bool initialized_ = false;
  std::int64_t lastThreadIdx_ = 0;
};

LlmDataLoaderImpl::~LlmDataLoaderImpl() {
  shutDown_->store(true);
  for (const auto &b : startBarrier_) {
    b->arrive_and_wait();
  }
}
int64_t LlmDataLoaderImpl::iterations_per_epoch() const {
  return dataset_.size() / (batchSize * world_size);
}

HostBuffer::~HostBuffer() { CS_CHECK_CUDART(cudaFreeHost(ptr)); }

const std::shared_ptr<DataLoader::Impl> &DataLoader::impl() const {
  return impl_;
}

int64_t DataLoader::iterations_per_epoch() const {
  return impl()->iterations_per_epoch();
}

void threadLoaderTask(const std::span<const std::string> files,
                      const int64_t batchSize, const int64_t startIter,
                      const int64_t totalIters,
                      const std::shared_ptr<HostBuffer[]> buffer,
                      const Event event,
                      const std::shared_ptr<std::atomic<bool>> shutDown,
                      const std::shared_ptr<std::barrier<>> startBarrier,
                      const std::shared_ptr<std::barrier<>> endBarrier) {
  startBarrier->arrive_and_wait();
  const auto filesystem = std::make_shared<arrow::fs::LocalFileSystem>();
  const auto format = std::make_shared<arrow::dataset::ParquetFileFormat>();
  auto result = arrow::dataset::FileSystemDatasetFactory::Make(
      filesystem, std::vector<std::string>{files.begin(), files.end()}, format,
      {});
  CS_ASSERT_TRUE(result.ok(), fmt::format("Failed to make dataset factory: {}",
                                          result.status().ToString()));
  const auto factory = result.ValueOrDie();
  const auto dataset_result = factory->Finish();
  CS_ASSERT_TRUE(dataset_result.ok(),
                 fmt::format("Failed to finish dataset: {}",
                             dataset_result.status().ToString()));
  const auto &dataset = dataset_result.ValueOrDie();
  auto scanner_builder_result = dataset->NewScan();
  CS_ASSERT_TRUE(scanner_builder_result.ok(),
                 fmt::format("Failed to start a new scan: {}",
                             scanner_builder_result.status().ToString()));

  const auto scanner_builder = scanner_builder_result.ValueOrDie();
  scanner_builder->GetScanOptions().ValueOrDie()->batch_size = batchSize;

  const auto project_status = scanner_builder->Project({"input_ids", "labels"});
  CS_ASSERT_TRUE(project_status.ok(),
                 fmt::format("Failed to set projection columns: {}",
                             project_status.ToString()));

  const auto scanner_result = scanner_builder->Finish();
  CS_ASSERT_TRUE(scanner_result.ok(),
                 fmt::format("Failed to create scanner: {}",
                             scanner_result.status().ToString()));
  const auto &scanner = scanner_result.ValueOrDie();
  scanner->options()->batch_readahead = 0;
  auto it = scanner->ScanBatches().ValueOrDie();
  auto batch = it.Next().ValueOrDie();
  for (int64_t i = 1; i < startIter; ++i) {
    batch = it.Next().ValueOrDie();
  }
  int64_t itCount = 1;

  const auto rows = batchSize;
  const auto cols =
      std::static_pointer_cast<arrow::ListArray>(batch.record_batch->column(0))
          ->value_length(0);
  // TODO(Jie): maybe padding
  const auto ld = cols;

  for (int i = 0; i < 2; ++i) {
    buffer[i].options = buffer[i].options.dtype(at::kLong);
    const auto size = sizeof(std::int64_t) * rows * ld;
    if (buffer[i].size < size) {
      CS_CHECK_CUDART(cudaFreeHost(buffer[i].ptr));
      CS_CHECK_CUDART(cudaMallocHost(&buffer[i].ptr, size));
      buffer[i].size = size;
      buffer[i].validDataSize = size;
    }
    buffer[i].sizes = {rows, cols};
  }

  while (true) {
    for (int i = 0; i < 2; ++i) {
      auto listArray = std::static_pointer_cast<arrow::ListArray>(
          batch.record_batch->column(i));
      if (listArray->value_type()->id() == arrow::Type::INT32) {
        for (int j = 0; j < listArray->length(); j++) {
          auto subArray = std::static_pointer_cast<arrow::Int32Array>(
              listArray->values()->Slice(listArray->value_offset(j),
                                         listArray->value_length(j)));
          for (int k = 0; k < subArray->length(); k++) {
            static_cast<int64_t *>(buffer[i].ptr)[k + j * cols] =
                subArray->Value(k);
          }
        }
      } else if (listArray->value_type()->id() == arrow::Type::INT64) {
        for (int j = 0; j < listArray->length(); j++) {
          auto subArray = std::static_pointer_cast<arrow::Int64Array>(
              listArray->values()->Slice(listArray->value_offset(j),
                                         listArray->value_length(j)));
          for (int k = 0; k < subArray->length(); k++) {
            static_cast<int64_t *>(buffer[i].ptr)[k + j * cols] =
                subArray->Value(k);
          }
        }
      } else {
        CS_ASSERT_TRUE(false,
                       fmt::format("Not supported dataset value type: {}",
                                   listArray->value_type()->ToString()));
      }
    }
    startBarrier->arrive_and_wait();
    if (shutDown->load()) {
      break;
    }
    endBarrier->arrive_and_wait();
    event.synchronize();
    ++itCount;
    if (itCount >= totalIters) {
      itCount = 0;
      it = scanner->ScanBatches().ValueOrDie();
      for (int64_t i = 0; i < startIter; ++i) {
        (void)it.Next();
      }
    } else {
      batch = it.Next().ValueOrDie();
    }
  }
}

std::unordered_map<std::string, Tensor> LlmDataLoader::load(
    const Scheduler &scheduler) const {
  const auto implCast = std::dynamic_pointer_cast<LlmDataLoaderImpl>(impl());

  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* x, y */, const Event &event,
                  const std::span<HostBuffer> buffer)
        : Task::Impl{std::move(output), {}, kMain, kLoader},
          event{event},
          buffer{buffer} {}
    const Event event;
    const std::span<HostBuffer> buffer;

    void operator()() const override {
      const auto stream = c10::cuda::getCurrentCUDAStream();
      const auto device = stream.device();
      CS_ASSERT_TRUE(output().size() == buffer.size(), "Incorrect buffer size");
      for (std::size_t i = 0; i < buffer.size(); ++i) {
        output()[i].impl()->tensor() =
            at::empty(buffer[i].sizes, buffer[i].options.device(device));
        AT_DISPATCH_FLOATING_TYPES_AND3(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            at::ScalarType::Long, output()[i].impl()->tensor().scalar_type(),
            "memcpy", [&] {
              using T = scalar_t;
              CS_CHECK_CUDART(cudaMemcpyAsync(
                  output()[i].impl()->tensor().data_ptr(), buffer[i].ptr,
                  sizeof(T) * output()[i].impl()->tensor().numel(),
                  cudaMemcpyHostToDevice, stream.stream()));
            });
      }
      event.record();
    }
    [[nodiscard]] const char *name() const override {
      return "cs::LlmDataLoader::load";
    }
  };

  std::vector<Tensor> output;
  output.resize(2);
  std::unordered_map<std::string, Tensor> map;
  map.insert({"input_ids", output[0]});
  map.insert({"labels", output[1]});

  implCast->startBarrier_[implCast->lastThreadIdx_]->arrive_and_wait();
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{output,
           implCast->events_[implCast->lastThreadIdx_],
           {implCast->buffers_[implCast->lastThreadIdx_].get(),
            implCast->buffers_[implCast->lastThreadIdx_].get() + 2}})});
  implCast->endBarrier_[implCast->lastThreadIdx_]->arrive_and_wait();
  ++implCast->lastThreadIdx_;
  implCast->lastThreadIdx_ =
      implCast->lastThreadIdx_ % implCast->threadVector_.size();

  return map;
}

LlmDataLoader::LlmDataLoader(const LlmDataset &dataset, int64_t batchSize,
                             int64_t numWorkers, bool shuffle) {
  CS_ASSERT_TRUE(shuffle == false, "We do not support shuffle now");
  auto rank = 0;
  auto worldSize = 1;
  const auto impl = std::make_shared<LlmDataLoaderImpl>(
      dataset, batchSize, numWorkers, shuffle, rank, worldSize);
  impl_ = impl;
  impl->shutDown_ = std::make_shared<std::atomic<bool>>(false);
  impl->startBarrier_.reserve(numWorkers);
  impl->endBarrier_.reserve(numWorkers);
  impl->threadVector_.reserve(numWorkers);
  impl->events_.reserve(numWorkers);
  impl->buffers_.reserve(numWorkers);
  const auto batchEachProcess = impl->dataset_.size() / worldSize;
  const auto batchOffsetOfThisProcess = batchEachProcess * rank;
  auto &fileOffsets = impl->dataset_.impl()->file_offsets();
  size_t startFileIdxOfThisProcess;
  for (startFileIdxOfThisProcess = 0;
       startFileIdxOfThisProcess < fileOffsets.size();
       ++startFileIdxOfThisProcess) {
    if (fileOffsets[startFileIdxOfThisProcess + 1] > batchOffsetOfThisProcess) {
      break;
    }
  }
  CS_ASSERT_TRUE(startFileIdxOfThisProcess != fileOffsets.size(),
                 "Wrong config, maybe batch size is too large");
  const auto batchEachWorker = batchEachProcess / numWorkers;
  const int64_t totalIterPerWorker = batchEachProcess / batchSize / numWorkers;
  CS_ASSERT_TRUE(
      numWorkers <= static_cast<int64_t>(impl->dataset_.files().size()),
      "Too many workers");
  for (int64_t i = 0; i < numWorkers; ++i) {
    impl->startBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
    impl->endBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
    impl->events_.emplace_back();
    impl->buffers_.emplace_back(new HostBuffer[2]);
    const auto batchOffsetOfThisThread =
        batchOffsetOfThisProcess + i * batchEachWorker;
    size_t startFileIdxOfThisThread;
    for (startFileIdxOfThisThread = startFileIdxOfThisProcess;
         startFileIdxOfThisThread < fileOffsets.size();
         ++startFileIdxOfThisThread) {
      if (fileOffsets[startFileIdxOfThisThread + 1] > batchOffsetOfThisThread) {
        break;
      }
    }
    CS_ASSERT_TRUE(startFileIdxOfThisThread != fileOffsets.size(),
                   "Internal error");
    const int64_t startIter =
        (batchOffsetOfThisThread - fileOffsets[startFileIdxOfThisThread]) /
        batchSize;
    size_t endFileIdx;
    for (endFileIdx = startFileIdxOfThisThread; endFileIdx < fileOffsets.size();
         ++endFileIdx) {
      if (fileOffsets[endFileIdx + 1] >=
          batchOffsetOfThisThread + batchEachWorker) {
        break;
      }
    }
    impl->threadVector_.emplace_back(
        threadLoaderTask,
        std::span{impl->dataset_.files().data() + startFileIdxOfThisThread,
                  impl->dataset_.files().data() + endFileIdx + 1},
        batchSize, startIter, totalIterPerWorker, impl->buffers_[i],
        impl->events_[i], impl->shutDown_, impl->startBarrier_[i],
        impl->endBarrier_[i]);

    impl->startBarrier_[i]->arrive_and_wait();
  }
  impl->lastThreadIdx_ = 0;
}

LlmDataLoader::LlmDataLoader(const LlmDataset &dataset,
                             const communication::Comm &comm, int64_t batchSize,
                             int64_t numWorkers, bool shuffle) {
  CS_ASSERT_TRUE(shuffle == false, "We do not support shuffle now");
  auto rank = comm.get_rank();
  auto worldSize = comm.get_size();
  const auto impl = std::make_shared<LlmDataLoaderImpl>(
      dataset, batchSize, numWorkers, shuffle, rank, worldSize);
  impl_ = impl;
  impl->shutDown_ = std::make_shared<std::atomic<bool>>(false);
  impl->startBarrier_.reserve(numWorkers);
  impl->endBarrier_.reserve(numWorkers);
  impl->threadVector_.reserve(numWorkers);
  impl->events_.reserve(numWorkers);
  impl->buffers_.reserve(numWorkers);
  const auto batchEachProcess = impl->dataset_.size() / worldSize;
  const auto batchOffsetOfThisProcess = batchEachProcess * rank;
  auto &fileOffsets = impl->dataset_.impl()->file_offsets();
  size_t startFileIdxOfThisProcess;
  for (startFileIdxOfThisProcess = 0;
       startFileIdxOfThisProcess < fileOffsets.size();
       ++startFileIdxOfThisProcess) {
    if (fileOffsets[startFileIdxOfThisProcess + 1] > batchOffsetOfThisProcess) {
      break;
    }
  }
  CS_ASSERT_TRUE(startFileIdxOfThisProcess != fileOffsets.size(),
                 "Wrong config, maybe batch size is too large");
  const auto batchEachWorker = batchEachProcess / numWorkers;
  const int64_t totalIterPerWorker = batchEachProcess / batchSize / numWorkers;
  for (int64_t i = 0; i < numWorkers; ++i) {
    impl->startBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
    impl->endBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
    impl->events_.emplace_back();
    impl->buffers_.emplace_back(new HostBuffer[2]);
    const auto batchOffsetOfThisThread =
        batchOffsetOfThisProcess + i * batchEachWorker;
    size_t startFileIdxOfThisThread;
    for (startFileIdxOfThisThread = startFileIdxOfThisProcess;
         startFileIdxOfThisThread < fileOffsets.size();
         ++startFileIdxOfThisThread) {
      if (fileOffsets[startFileIdxOfThisThread + 1] > batchOffsetOfThisThread) {
        break;
      }
    }
    CS_ASSERT_TRUE(startFileIdxOfThisThread != fileOffsets.size(),
                   "Internal error");
    const int64_t startIter =
        (batchOffsetOfThisThread - fileOffsets[startFileIdxOfThisThread]) /
        batchSize;
    size_t endFileIdx;
    for (endFileIdx = startFileIdxOfThisThread; endFileIdx < fileOffsets.size();
         ++endFileIdx) {
      if (fileOffsets[endFileIdx + 1] >=
          batchOffsetOfThisProcess + batchEachWorker) {
        break;
      }
    }
    impl->threadVector_.emplace_back(
        threadLoaderTask,
        std::span{impl->dataset_.files().data() + startFileIdxOfThisThread,
                  impl->dataset_.files().data() + endFileIdx + 1},
        batchSize, startIter, totalIterPerWorker, impl->buffers_[i],
        impl->events_[i], impl->shutDown_, impl->startBarrier_[i],
        impl->endBarrier_[i]);

    impl->startBarrier_[i]->arrive_and_wait();
  }
  impl->lastThreadIdx_ = 0;
}
}  // namespace cs::data
