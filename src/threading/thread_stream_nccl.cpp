#include "threading/thread_stream_nccl.h"

#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>

#include "logger.h"

namespace dllm {
namespace {
void setThreadAffinity(std::thread &th, int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);

  int rc =
      pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "core binding error with code {}", rc);
  }
}

void threadTask(const ncclUniqueId id, const int ncclWorldSize,
                const int ncclRank, const int deviceRank,
                std::queue<TaskNccl> *taskQueue, std::mutex *queueMutex,
                std::condition_variable *cv, std::mutex *cvMutex,
                std::atomic<bool> *shutDown) {
  ContextNccl context;
  CHECK_CUDART(cudaSetDevice(deviceRank));
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  CHECK_NCCL(ncclCommInitRank(&context.ncclComm, ncclWorldSize, id, ncclRank));
  context.ncclRank = ncclRank;
  while (!shutDown->load()) {
    TaskNccl task;
    std::unique_lock lock{*queueMutex};
    if (!taskQueue->empty()) {
      task = std::move(taskQueue->front());
      taskQueue->pop();
    }
    lock.unlock();
    if (task.valid()) {
      task(&context);
    } else {
      std::unique_lock<std::mutex> uniqueLock{*cvMutex};
      cv->wait(uniqueLock,
               [&] { return shutDown->load() || !taskQueue->empty(); });
    }
  }
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

std::shared_ptr<FutureNccl> ThreadStreamNccl::submit(TaskNccl &&task) {
  auto future = task.get_future();
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
  return std::make_shared<FutureNccl>(std::move(future));
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
