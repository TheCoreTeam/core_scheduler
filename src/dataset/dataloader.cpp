#include "dataset/dataloader.h"

#include <ATen/Dispatch.h>

#include <pcg_random.hpp>
#include <queue>

#include "dataset/dataset.h"
#include "internal_utils.h"
#include "logger.h"
#include "random.h"
#include "tensor_friend.h"
#include "threading/task_cudart.h"

namespace arrow {
class ListArray;
}

namespace dllm::random {
RandomState &getRandomState();
}

namespace dllm::dataset {
struct LlmDataset::RowAccessor::Impl {
  const std::shared_ptr<const arrow::ListArray> inputIdsRow;
  const std::int64_t inputIdsRowOffset;
  const std::shared_ptr<const arrow::ListArray> targetsRow;
  const std::int64_t targetsRowOffset;

  [[nodiscard]] LlmDataset::Element accessCol(const std::int64_t colIdx) const;

  [[nodiscard]] std::int64_t cols() const;
};

struct LlmDataLoader::Impl {
  Impl(const std::shared_ptr<const LlmDataset> &dataset, int localRank,
       int batch_size, int num_workers, bool shuffle,
       const std::vector<int> &bindingMap);

  ~Impl() {
    shutDown = true;
    cv.notify_all();
    for (auto &t : threadVector) {
      while (!t.joinable()) {
        cv.notify_all();
      }
      t.join();
    }
  }

  struct HostBuffer {
    void *ptr{nullptr};
    std::size_t size{0};
    std::size_t validDataSize{0};

    ~HostBuffer() { CHECK_CUDART(cudaFreeHost(ptr)); }
  };

  using TaskLoader = std::packaged_task<void(
      const ContextCudart *, const HostBuffer *, const HostBuffer *)>;

  void submit(TaskLoader &&task) const;

  const std::shared_ptr<const LlmDataset> dataset;
  const int batch_size;
  const int num_workers;
  const bool shuffle;
  std::vector<std::jthread> threadVector{};
  mutable std::queue<TaskLoader> taskQueue{};
  mutable std::mutex queueMutex{};
  mutable std::condition_variable cv{};
  mutable std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
};

namespace {
void setThreadAffinity(std::jthread &th, const int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);

  const int rc =
      pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
  DLLM_ASSERT_TRUE(rc == 0, "core binding error with code {}", rc);
}

template <typename Impl, typename HostBuffer, typename TaskLoader>
void threadTask(const Impl *self, const int localRank) {
  ContextCudart context{.deviceRank = localRank};
  CHECK_CUDART(cudaSetDevice(localRank));
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  HostBuffer xBuffer{}, yBuffer{};

  auto prepareData = [&] {
    auto &[seed, offset] = random::getRandomState();
    const auto rows = self->dataset->rows();
    const auto cols = self->dataset->cols();
    // TODO(Jie): maybe padding
    const auto ld = cols;
    pcg64 rng(seed);
    rng.advance(offset.fetch_add(rows));

    const auto xSize = sizeof(std::int64_t) * rows * ld;
    const auto ySize = sizeof(std::int64_t) * rows * ld;
    if (xBuffer.size < xSize) {
      CHECK_CUDART(cudaFreeHost(xBuffer.ptr));
      CHECK_CUDART(cudaMallocHost(&xBuffer.ptr, xSize));
      xBuffer.size = xSize;
    }
    if (yBuffer.size < ySize) {
      CHECK_CUDART(cudaFreeHost(yBuffer.ptr));
      CHECK_CUDART(cudaMallocHost(&yBuffer.ptr, ySize));
      yBuffer.size = ySize;
    }

    xBuffer.validDataSize = xSize;
    yBuffer.validDataSize = ySize;

    const auto xPtr = static_cast<std::int64_t *>(xBuffer.ptr);
    const auto yPtr = static_cast<std::int64_t *>(yBuffer.ptr);

    std::uniform_int_distribution<int64_t> distribution(
        0, self->dataset->rows() - 1);
    for (std::int64_t r = 0; r < rows; ++r) {
      const auto rowAccessor = self->dataset->accessRow(distribution(rng));
      for (std::int64_t c = 0; c < cols; ++c) {
        const auto element = rowAccessor.accessCol(c);
        xPtr[r * ld + c] = element.input_id;
        yPtr[r * ld + c] = element.target;
      }
    }
  };

  prepareData();
  while (!self->shutDown.load()) {
    TaskLoader task;
    std::unique_lock lock{self->queueMutex};
    if (!self->taskQueue.empty()) {
      task = std::move(self->taskQueue.front());
      self->taskQueue.pop();
    }
    lock.unlock();
    if (task.valid()) {
      task(&context, &xBuffer, &yBuffer);
      task = {};
      prepareData();
    } else {
      std::unique_lock<std::mutex> uniqueLock{self->cvMutex};
      self->cv.wait(uniqueLock, [&] {
        return self->shutDown.load() || !self->taskQueue.empty();
      });
    }
  }
  std::unique_lock lock{self->queueMutex};
  self->taskQueue = {};
  lock.unlock();
  CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
}
}  // namespace

LlmDataLoader::Impl::Impl(const std::shared_ptr<const LlmDataset> &dataset,
                          const int localRank, const int batch_size,
                          const int num_workers, const bool shuffle,
                          const std::vector<int> &bindingMap)
    : dataset{dataset},
      batch_size{batch_size},
      num_workers{num_workers},
      shuffle{shuffle} {
  if (!bindingMap.empty() &&
      bindingMap.size() != static_cast<std::size_t>(num_workers)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "bindingMap size incorrect");
  }
  threadVector.reserve(num_workers);
  for (int i = 0; i < num_workers; ++i) {
    threadVector.emplace_back(threadTask<Impl, HostBuffer, TaskLoader>, this,
                              localRank);
  }
  if (!bindingMap.empty()) {
    for (int i = 0; i < num_workers; ++i) {
      setThreadAffinity(threadVector[i], bindingMap[i]);
    }
  }
}

LlmDataLoader::LlmDataLoader(const std::shared_ptr<const LlmDataset> &dataset,
                             const int localRank, const int batch_size,
                             const int num_workers, const bool shuffle,
                             const std::vector<int> &bindingMap)
    : impl_{std::make_unique<Impl>(dataset, localRank, batch_size, num_workers,
                                   shuffle, bindingMap)} {}

void LlmDataLoader::fill(const std::shared_ptr<Tensor> &x,
                         const std::shared_ptr<Tensor> &y) const {
  auto task = Impl::TaskLoader{[x = x, y = y, xFuture = x->future(),
                                yFuture = y->future()](
                                   const ContextCudart *context,
                                   const Impl::HostBuffer *xBuffer,
                                   const Impl::HostBuffer *yBuffer) mutable {
    DLLM_ASSERT_TRUE(x->numel() == static_cast<int64_t>(xBuffer->validDataSize),
                     "Tensor size mismatch with buffer size");
    DLLM_ASSERT_TRUE(y->numel() == static_cast<int64_t>(yBuffer->validDataSize),
                     "Tensor size mismatch with buffer size");
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        DLLM_EXTRACT_TENSOR(x).scalar_type(), "memcpy", [&] {
          using T = scalar_t;
          util::FutureGuard xGuard{xFuture};
          CHECK_CUDART(
              cudaMemcpyAsync(DLLM_EXTRACT_TENSOR(x).data_ptr(), xBuffer->ptr,
                              sizeof(T) * DLLM_EXTRACT_TENSOR(x).numel(),
                              cudaMemcpyHostToDevice, context->cudaStream));
        });
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        DLLM_EXTRACT_TENSOR(y).scalar_type(), "memcpy", [&] {
          using T = scalar_t;
          util::FutureGuard xGuard{xFuture};
          CHECK_CUDART(
              cudaMemcpyAsync(DLLM_EXTRACT_TENSOR(y).data_ptr(), xBuffer->ptr,
                              sizeof(T) * DLLM_EXTRACT_TENSOR(y).numel(),
                              cudaMemcpyHostToDevice, context->cudaStream));
        });
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};

  const TaskFuture future = task.get_future();
  x->resetFuture(future);
  y->resetFuture(future);
  impl_->submit(std::move(task));
}

void LlmDataLoader::Impl::submit(TaskLoader &&task) const {
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
}
}  // namespace dllm::dataset
