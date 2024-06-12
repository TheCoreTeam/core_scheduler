#include "data/dataloader.h"

#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <barrier>
#include <future>
#include <pcg_random.hpp>
#include <queue>

#include "data/dataloader_impl.h"
#include "data/dataset.h"
#include "logger.h"
#include "random.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace arrow {
class ListArray;
}

namespace dllm::random {
RandomState &getRandomState();
}

namespace dllm::data {
inline DataLoader::Impl::Impl(const int batchSize, const int numWorkers,
                              const bool shuffle)
    : batchSize{batchSize}, numWorkers{numWorkers}, shuffle{shuffle} {}

struct LlmDataset::RowAccessor::Impl {
  const std::shared_ptr<const arrow::ListArray> inputIdsRow;
  const std::int64_t inputIdsRowOffset;
  const std::shared_ptr<const arrow::ListArray> targetsRow;
  const std::int64_t targetsRowOffset;

  [[nodiscard]] Element accessCol(std::int64_t colIdx) const;

  [[nodiscard]] std::int64_t cols() const;
};

struct LlmBuffer {
  struct HostBuffer {
    void *ptr{nullptr};
    std::size_t size{0};
    IntArray sizes{};
    TensorOptions options;
    std::size_t validDataSize{0};

    ~HostBuffer();
  } x, y;
};

struct LlmDataLoaderImpl final : DataLoader::Impl {
  LlmDataLoaderImpl(const LlmDataset &dataset, const int batch_size,
                    const int num_workers, const bool shuffle)
      : Impl{batch_size, num_workers, shuffle}, dataset{dataset} {}

  const LlmDataset dataset;

  [[nodiscard]] std::function<void(const std::shared_ptr<LlmBuffer> &)>
  getFiller(int64_t batchsize) const;

  ~LlmDataLoaderImpl() override;

  std::vector<std::shared_ptr<std::barrier<>>> startBarrier_{};
  std::vector<Event> events_{};
  std::vector<std::shared_ptr<LlmBuffer>> buffers_{};
  std::vector<std::shared_ptr<std::barrier<>>> endBarrier_{};
  std::vector<std::jthread> threadVector_{};
  std::shared_ptr<std::atomic<bool>> shutDown_{nullptr};
  bool initialized_ = false;
  std::size_t lastThreadIdx_ = 0;
};

std::function<void(const std::shared_ptr<LlmBuffer> &)>
LlmDataLoaderImpl::getFiller(int64_t batchsize) const {
  return std::function{[batchsize = batchsize, dataset = dataset](
                           const std::shared_ptr<LlmBuffer> &buffer) {
    auto &[seed, offset] = random::getRandomState();
    const auto rows = batchsize;
    const auto cols = dataset.cols();
    // TODO(Jie): maybe padding
    const auto ld = cols;
    pcg64 rng(seed);
    rng.advance(offset.fetch_add(rows));

    buffer->x.options = buffer->x.options.dtype(at::kLong);
    buffer->y.options = buffer->y.options.dtype(at::kLong);

    const auto xSize = sizeof(std::int64_t) * rows * ld;
    const auto ySize = sizeof(std::int64_t) * rows * ld;
    if (buffer->x.size < xSize) {
      CHECK_CUDART(cudaFreeHost(buffer->x.ptr));
      CHECK_CUDART(cudaMallocHost(&buffer->x.ptr, xSize));
      buffer->x.size = xSize;
    }
    if (buffer->y.size < ySize) {
      CHECK_CUDART(cudaFreeHost(buffer->y.ptr));
      CHECK_CUDART(cudaMallocHost(&buffer->y.ptr, ySize));
      buffer->y.size = ySize;
    }

    buffer->x.validDataSize = xSize;
    buffer->y.validDataSize = ySize;

    const auto xPtr = static_cast<std::int64_t *>(buffer->x.ptr);
    const auto yPtr = static_cast<std::int64_t *>(buffer->y.ptr);

    std::uniform_int_distribution<int64_t> distribution(0, dataset.rows() - 1);
    for (std::int64_t r = 0; r < rows; ++r) {
      const auto rowAccessor = dataset.accessRow(distribution(rng));
      for (std::int64_t c = 0; c < cols; ++c) {
        const auto [input_id, target] = rowAccessor.accessCol(c);
        xPtr[r * ld + c] = input_id;
        yPtr[r * ld + c] = target;
      }
    }
    buffer->x.sizes = {rows, cols};
    buffer->y.sizes = {rows, cols};
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

LlmBuffer::HostBuffer::~HostBuffer() { CHECK_CUDART(cudaFreeHost(ptr)); }

const std::shared_ptr<DataLoader::Impl> &DataLoader::impl() const {
  return impl_;
}

void threadLoaderTask(
    const std::function<void(const std::shared_ptr<LlmBuffer> &)> filler,
    const std::shared_ptr<LlmBuffer> buffer, const Event event,
    const std::shared_ptr<std::atomic<bool>> shutDown,
    const std::shared_ptr<std::barrier<>> startBarrier,
    const std::shared_ptr<std::barrier<>> endBarrier) {
  startBarrier->arrive_and_wait();
  while (true) {
    filler(buffer);
    startBarrier->arrive_and_wait();
    if (shutDown->load()) {
      break;
    }
    endBarrier->arrive_and_wait();
    event.synchronize();
  }
}

void LlmDataLoader::load(const Scheduler &scheduler, Tensor &x,
                         Tensor &y) const {
  const auto implCast = std::dynamic_pointer_cast<LlmDataLoaderImpl>(impl());

  if (!implCast->initialized_) [[unlikely]] {
    // reset things
    if (implCast->shutDown_ != nullptr) {
      implCast->shutDown_->store(true);
      for (const auto &b : implCast->startBarrier_) {
        (void)b->arrive();
      }
      implCast->startBarrier_.clear();
      for (const auto &b : implCast->endBarrier_) {
        (void)b->arrive();
      }
      implCast->endBarrier_.clear();
      implCast->threadVector_.clear();
      implCast->events_.clear();
      implCast->buffers_.clear();
    }
    implCast->shutDown_ = std::make_shared<std::atomic<bool>>(false);
    implCast->startBarrier_.reserve(implCast->numWorkers);
    implCast->endBarrier_.reserve(implCast->numWorkers);
    implCast->threadVector_.reserve(implCast->numWorkers);
    implCast->events_.reserve(implCast->numWorkers);
    implCast->buffers_.reserve(implCast->numWorkers);
    for (int8_t i = 0; i < implCast->numWorkers; ++i) {
      implCast->startBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
      implCast->endBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
      implCast->events_.emplace_back();
      implCast->buffers_.emplace_back(std::make_shared<LlmBuffer>());
      implCast->threadVector_.emplace_back(
          threadLoaderTask, implCast->getFiller(implCast->batchSize),
          implCast->buffers_[i], implCast->events_[i], implCast->shutDown_,
          implCast->startBarrier_[i], implCast->endBarrier_[i]);
      implCast->startBarrier_[i]->arrive_and_wait();
    }
    implCast->lastThreadIdx_ = 0;
  }

  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* x, y */, const Event &event,
                  std::shared_ptr<LlmBuffer> buffer)
        : Task::Impl{std::move(output), {}, loader},
          event{event},
          buffer{std::move(buffer)} {}
    const Event event;
    const std::shared_ptr<LlmBuffer> buffer;

    void operator()() const override {
      const auto stream = c10::cuda::getCurrentCUDAStream();
      const auto device = stream.device();
      output()[0].impl()->tensor() =
          at::empty(buffer->x.sizes, buffer->x.options.device(device));
      AT_DISPATCH_FLOATING_TYPES_AND3(
          at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Long,
          output()[0].impl()->tensor().scalar_type(), "memcpy", [&] {
            using T = scalar_t;
            CHECK_CUDART(cudaMemcpyAsync(
                output()[0].impl()->tensor().data_ptr(), buffer->x.ptr,
                sizeof(T) * output()[0].impl()->tensor().numel(),
                cudaMemcpyHostToDevice, stream.stream()));
          });
      output()[1].impl()->tensor() =
          at::empty(buffer->y.sizes, buffer->y.options.device(device));
      AT_DISPATCH_FLOATING_TYPES_AND3(
          at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Long,
          output()[1].impl()->tensor().scalar_type(), "memcpy", [&] {
            using T = scalar_t;
            CHECK_CUDART(cudaMemcpyAsync(
                output()[1].impl()->tensor().data_ptr(), buffer->y.ptr,
                sizeof(T) * output()[1].impl()->tensor().numel(),
                cudaMemcpyHostToDevice, stream.stream()));
          });
      event.record();
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::LlmDataLoader::load";
    }
  };

  x = Tensor{};
  y = Tensor{};

  implCast->startBarrier_[implCast->lastThreadIdx_]->arrive_and_wait();
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{x, y},
           implCast->events_[implCast->lastThreadIdx_],
           implCast->buffers_[implCast->lastThreadIdx_]})});
  implCast->endBarrier_[implCast->lastThreadIdx_]->arrive_and_wait();
  ++implCast->lastThreadIdx_;
  implCast->lastThreadIdx_ =
      implCast->lastThreadIdx_ % implCast->threadVector_.size();
}

LlmDataLoader::LlmDataLoader(const LlmDataset &dataset, int batchSize,
                             int numWorkers, bool shuffle) {
  impl_ = std::make_shared<LlmDataLoaderImpl>(dataset, batchSize, numWorkers,
                                              shuffle);
}
}  // namespace dllm::data
