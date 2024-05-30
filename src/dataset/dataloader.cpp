#include "dataset/dataloader.h"

#include <pcg_random.hpp>
#include <queue>

#include "dataset/dataset.h"
#include "random/random_internal.h"
#include "threading/task_cudart.h"
#include "util.h"

namespace dllm::dataset {
namespace {
void setThreadAffinity(std::thread &th, const int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);

  const int rc =
      pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
  DLLM_ASSERT_TRUE(rc == 0, "core binding error with code {}", rc);
}

struct LlmDataLoaderImpl {
  const std::shared_ptr<const LlmDataset> dataset;
  const int batch_size;
  const int num_workers;
  const bool shuffle;

  LlmDataLoaderImpl(const std::shared_ptr<const LlmDataset> &dataset,
                    const int localRank, const int batch_size,
                    const int num_workers, const bool shuffle,
                    const std::vector<int> &bindingMap)
      : dataset{dataset},
        batch_size{batch_size},
        num_workers{num_workers},
        shuffle{shuffle} {
    if (!bindingMap.empty() && bindingMap.size() != num_workers) {
      SPDLOG_LOGGER_CRITICAL(&logger(), "bindingMap size incorrect");
    }
    threadVector.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      threadVector.emplace_back(threadTask, this, localRank);
    }
    if (!bindingMap.empty()) {
      for (int i = 0; i < num_workers; ++i) {
        setThreadAffinity(threadVector[i], bindingMap[i]);
      }
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

  std::vector<std::thread> threadVector{};
  mutable std::queue<TaskLoader> taskQueue{};
  mutable std::mutex queueMutex{};
  mutable std::condition_variable cv{};
  mutable std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};

  ~LlmDataLoaderImpl() {
    shutDown = true;
    cv.notify_all();
    for (auto &t : threadVector) {
      while (!t.joinable()) {
        cv.notify_all();
      }
      t.join();
    }
  }

  void fill(const std::shared_ptr<Tensor2D> &x,
            const std::shared_ptr<Tensor2D> &y) const {
    auto task =
        TaskLoader{[x = x, y = y, xFuture = *x->future, yFuture = *y->future](
                       const ContextCudart *context, const HostBuffer *xBuffer,
                       const HostBuffer *yBuffer) mutable {
          if (cute::cosize(x->layout) != xBuffer->validDataSize) {
            SPDLOG_LOGGER_CRITICAL(&logger(),
                                   "Tensor size mismatch with buffer size");
          }
          if (cute::cosize(y->layout) != yBuffer->validDataSize) {
            SPDLOG_LOGGER_CRITICAL(&logger(),
                                   "Tensor size mismatch with buffer size");
          }
          {
            util::FutureGuard xRGuard{xFuture.rFuture};
            util::FutureGuard xWGuard{xFuture.wFuture};
            CHECK_CUDART(cudaMemcpyAsync(
                x->data(), xBuffer->ptr, cute::cosize(x->layout),
                cudaMemcpyHostToDevice, context->cudaStream));
          }
          {
            util::FutureGuard yRGuard{yFuture.rFuture};
            util::FutureGuard yWGuard{yFuture.wFuture};
            CHECK_CUDART(cudaMemcpyAsync(
                y->data(), yBuffer->ptr, cute::cosize(y->layout),
                cudaMemcpyHostToDevice, context->cudaStream));
          }
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }};

    const TaskFuture future = task.get_future();
    x->future->wFuture = future;
    y->future->wFuture = future;
    submit(std::move(task));
  }

  void submit(TaskLoader &&task) const {
    std::unique_lock lock{queueMutex};
    taskQueue.push(std::move(task));
    lock.unlock();
    cv.notify_one();
  }

  static void threadTask(const LlmDataLoaderImpl *self, const int localRank) {
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
          const auto element = rowAccessor->accessCol(c);
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
};
}  // namespace

std::shared_ptr<const LlmDataLoader> LlmDataLoader::create(
    const std::shared_ptr<const LlmDataset> &dataset, const int localRank,
    const int batch_size, const int num_workers, const bool shuffle,
    const std::vector<int> &bindingMap) {
  return std::reinterpret_pointer_cast<const LlmDataLoader>(
      std::make_shared<const LlmDataLoaderImpl>(
          dataset, localRank, batch_size, num_workers, shuffle, bindingMap));
}

LlmDataLoader::~LlmDataLoader() {
  const auto impl = reinterpret_cast<LlmDataLoaderImpl *>(this);
  impl->~LlmDataLoaderImpl();
}

void LlmDataLoader::fill(const std::shared_ptr<Tensor2D> &x,
                         const std::shared_ptr<Tensor2D> &y) const {
  reinterpret_cast<const LlmDataLoaderImpl *>(this)->fill(x, y);
}
}  // namespace dllm::dataset
