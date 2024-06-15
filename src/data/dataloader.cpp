#include "data/dataloader.h"

#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <barrier>
#include <future>
#include <queue>
#include <span>

#include "data/dataloader_impl.h"
#include "data/dataset.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace arrow {
class ListArray;
}

namespace dllm::data {
inline DataLoader::Impl::Impl(const int64_t batchSize, const int64_t numWorkers,
                              const bool shuffle, const int64_t rank,
                              const int64_t worldSize)
    : batchSize{batchSize},
      numWorkers{numWorkers},
      shuffle{shuffle},
      rank{rank},
      worldSize{worldSize} {}

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
        dataset{dataset} {}

  const LlmDataset dataset;

  [[nodiscard]] std::function<void(const std::span<HostBuffer> &, int64_t)>
  getFiller(int64_t batchsize) const;

  ~LlmDataLoaderImpl() override;

  [[nodiscard]] int64_t iterationsPerEpoch() const override;

  std::vector<std::shared_ptr<std::barrier<>>> startBarrier_{};
  std::vector<Event> events_{};
  std::vector<std::shared_ptr<HostBuffer[]>> buffers_{};
  std::vector<std::shared_ptr<std::barrier<>>> endBarrier_{};
  std::vector<std::jthread> threadVector_{};
  std::shared_ptr<std::atomic<bool>> shutDown_{nullptr};
  bool initialized_ = false;
  std::int64_t lastThreadIdx_ = 0;
};

std::function<void(const std::span<HostBuffer> &, int64_t)>
LlmDataLoaderImpl::getFiller(int64_t batchsize) const {
  return std::function{
      [batchsize = batchsize, dataset = dataset](
          const std::span<HostBuffer> &buffer, const int64_t iteration) {
        const auto rows = batchsize;
        const auto cols = dataset.cols();
        // TODO(Jie): maybe padding
        const auto ld = cols;

        for (auto &i : buffer) {
          i.options = i.options.dtype(at::kLong);
          const auto size = sizeof(std::int64_t) * rows * ld;
          if (i.size < size) {
            CHECK_CUDART(cudaFreeHost(i.ptr));
            CHECK_CUDART(cudaMallocHost(&i.ptr, size));
            i.size = size;
            i.validDataSize = size;
          }
          i.sizes = {rows, cols};
        }
        std::vector<int64_t *> ptrs;
        std::vector<int64_t> lds;
        ptrs.reserve(buffer.size());
        for (const auto &i : buffer) {
          ptrs.push_back(static_cast<int64_t *>(i.ptr));
          lds.push_back(ld);
        }
        dataset.fillBatch(ptrs, lds, iteration * batchsize, batchsize);
      }};
}
LlmDataLoaderImpl::~LlmDataLoaderImpl() {
  shutDown_->store(true);
  for (const auto &b : startBarrier_) {
    (void)b->arrive();
  }
  for (const auto &b : endBarrier_) {
    (void)b->arrive();
  }
}
int64_t LlmDataLoaderImpl::iterationsPerEpoch() const {
  return dataset.rows() / (batchSize * worldSize);
}

HostBuffer::~HostBuffer() { CHECK_CUDART(cudaFreeHost(ptr)); }

const std::shared_ptr<DataLoader::Impl> &DataLoader::impl() const {
  return impl_;
}

int64_t DataLoader::iterationsPerEpoch() const {
  return impl()->iterationsPerEpoch();
}

void threadLoaderTask(
    const std::function<void(const std::span<HostBuffer> &, int64_t)> filler,
    const int64_t threadRank, const int64_t threadNum,
    const int64_t processRank, const int64_t worldSize,
    const int64_t iterations, const std::shared_ptr<HostBuffer[]> buffer,
    const int64_t bufferNum, const Event event,
    const std::shared_ptr<std::atomic<bool>> shutDown,
    const std::shared_ptr<std::barrier<>> startBarrier,
    const std::shared_ptr<std::barrier<>> endBarrier) {
  int64_t iteration = processRank + threadRank * worldSize;
  startBarrier->arrive_and_wait();
  while (true) {
    filler({buffer.get(), buffer.get() + bufferNum}, iteration);
    startBarrier->arrive_and_wait();
    if (shutDown->load()) {
      break;
    }
    endBarrier->arrive_and_wait();
    event.synchronize();
    iteration += threadNum * worldSize;
    if (iteration > iterations) {
      iteration = processRank + threadRank * worldSize;
    }
  }
}

std::unordered_map<std::string, Tensor> LlmDataLoader::load(
    const Scheduler &scheduler) const {
  const auto implCast = std::dynamic_pointer_cast<LlmDataLoaderImpl>(impl());

  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* x, y */, const Event &event,
                  const std::span<HostBuffer> buffer)
        : Task::Impl{std::move(output), {}, loader},
          event{event},
          buffer{buffer} {}
    const Event event;
    const std::span<HostBuffer> buffer;

    void operator()() const override {
      const auto stream = c10::cuda::getCurrentCUDAStream();
      const auto device = stream.device();
      DLLM_ASSERT_TRUE(output().size() == buffer.size(),
                       "Incorrect buffer size");
      for (std::size_t i = 0; i < buffer.size(); ++i) {
        output()[i].impl()->tensor() =
            at::empty(buffer[i].sizes, buffer[i].options.device(device));
        AT_DISPATCH_FLOATING_TYPES_AND3(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            at::ScalarType::Long, output()[i].impl()->tensor().scalar_type(),
            "memcpy", [&] {
              using T = scalar_t;
              CHECK_CUDART(cudaMemcpyAsync(
                  output()[i].impl()->tensor().data_ptr(), buffer[i].ptr,
                  sizeof(T) * output()[i].impl()->tensor().numel(),
                  cudaMemcpyHostToDevice, stream.stream()));
            });
      }
      event.record();
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::LlmDataLoader::load";
    }
  };

  std::vector<Tensor> output;
  const auto attributeNum = implCast->dataset.attributeNum();
  const auto &attributeNames = implCast->dataset.attributeNames();
  output.resize(attributeNum);
  std::unordered_map<std::string, Tensor> map;
  for (int i = 0; i < attributeNum; ++i) {
    map.insert({attributeNames[i], output[i]});
  }

  implCast->startBarrier_[implCast->lastThreadIdx_]->arrive_and_wait();
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{
      output,
      implCast->events_[implCast->lastThreadIdx_],
      {implCast->buffers_[implCast->lastThreadIdx_].get(),
       implCast->buffers_[implCast->lastThreadIdx_].get() + attributeNum}})});
  implCast->endBarrier_[implCast->lastThreadIdx_]->arrive_and_wait();
  ++implCast->lastThreadIdx_;
  implCast->lastThreadIdx_ =
      implCast->lastThreadIdx_ % implCast->threadVector_.size();

  return map;
}

LlmDataLoader::LlmDataLoader(const LlmDataset &dataset, int64_t batchSize,
                             int64_t numWorkers, bool shuffle, int64_t rank,
                             int64_t worldSize) {
  DLLM_ASSERT_TRUE(shuffle == false, "We do not support shuffle now");
  const auto impl = std::make_shared<LlmDataLoaderImpl>(
      dataset, batchSize, numWorkers, shuffle, rank, worldSize);
  impl_ = impl;
  impl->shutDown_ = std::make_shared<std::atomic<bool>>(false);
  impl->startBarrier_.reserve(numWorkers);
  impl->endBarrier_.reserve(numWorkers);
  impl->threadVector_.reserve(numWorkers);
  impl->events_.reserve(numWorkers);
  impl->buffers_.reserve(numWorkers);
  for (int8_t i = 0; i < numWorkers; ++i) {
    impl->startBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
    impl->endBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
    impl->events_.emplace_back();
    impl->buffers_.emplace_back(new HostBuffer[impl->dataset.attributeNum()]);
    impl->threadVector_.emplace_back(
        threadLoaderTask, impl->getFiller(batchSize), i, numWorkers, rank,
        worldSize, impl->iterationsPerEpoch(), impl->buffers_[i],
        impl->dataset.attributeNum(), impl->events_[i], impl->shutDown_,
        impl->startBarrier_[i], impl->endBarrier_[i]);
    impl->startBarrier_[i]->arrive_and_wait();
  }
  impl->lastThreadIdx_ = 0;
}
}  // namespace dllm::data
