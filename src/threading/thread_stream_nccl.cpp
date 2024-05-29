#include "threading/thread_stream_nccl.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>

#include "logger.h"

namespace dllm {
namespace {
void setThreadAffinity(std::thread &th, const int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);

  const int rc =
      pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "core binding error with code {}", rc);
  }
}

void threadTask(const ncclUniqueId id, const int ncclWorldSize,
                const int ncclRank, const int deviceRank,
                std::queue<TaskNccl> *taskQueue, std::mutex *queueMutex,
                std::condition_variable *cv, std::mutex *cvMutex,
                const std::atomic<bool> *shutDown) {
  ContextNccl context;
  CHECK_CUDART(cudaSetDevice(deviceRank));
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  CHECK_NCCL(ncclCommInitRank(&context.ncclComm, ncclWorldSize, id, ncclRank));
  context.ncclRank = ncclRank;
  context.commSize = ncclWorldSize;
  const auto stream = c10::cuda::getStreamFromExternal(
      context.cudaStream, static_cast<c10::DeviceIndex>(deviceRank));
  c10::cuda::CUDAStreamGuard streamGuard{stream};
  c10::cuda::CUDAGuard deviceGuard{static_cast<c10::DeviceIndex>(deviceRank)};

  while (!shutDown->load()) {
    TaskNccl task;
    std::unique_lock lock{*queueMutex};
    if (!taskQueue->empty()) {
      task = std::move(taskQueue->front());
      taskQueue->pop();
    }
    lock.unlock();
    if (task.valid()) {
      try {
        task(&context);
      } catch (const std::exception &e) {
        SPDLOG_LOGGER_CRITICAL(&::dllm::logger(), "Task failed with error: {}",
                               e.what());
        throw;
      }
      task = {};
    } else {
      std::unique_lock<std::mutex> uniqueLock{*cvMutex};
      cv->wait(uniqueLock,
               [&] { return shutDown->load() || !taskQueue->empty(); });
    }
  }
  std::unique_lock lock{*queueMutex};
  *taskQueue = {};
  lock.unlock();
  CHECK_NCCL(ncclCommDestroy(context.ncclComm));
  CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
}
}  // namespace

ThreadStreamNccl::~ThreadStreamNccl() {
  shutDown = true;
  cv.notify_one();
  while (!thread.joinable()) {
    cv.notify_one();
  }
  thread.join();
}

void ThreadStreamNccl::submit(TaskNccl &&task) {
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
}

ThreadStreamNccl::ThreadStreamNccl(const ncclUniqueId id,
                                   const int ncclWorldSize, const int ncclRank,
                                   const int deviceRank,
                                   const std::optional<const int> bindingMap)
    : thread{threadTask, id,          ncclWorldSize, ncclRank, deviceRank,
             &taskQueue, &queueMutex, &cv,           &cvMutex, &shutDown} {
  if (bindingMap.has_value()) {
    setThreadAffinity(thread, bindingMap.value());
  }
}
}  // namespace dllm
