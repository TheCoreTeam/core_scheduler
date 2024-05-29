#include "threading/thread_pool_cudart.h"

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

  int rc =
      pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "core binding error with code {}", rc);
  }
}

void threadTask(const int localRank, std::queue<TaskCudart> *taskQueue,
                std::mutex *queueMutex, std::condition_variable *cv,
                std::mutex *cvMutex, const std::atomic<bool> *shutDown) {
  ContextCudart context{.deviceRank = localRank};
  CHECK_CUDART(cudaSetDevice(localRank));
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  while (!shutDown->load()) {
    TaskCudart task;
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
  CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
}
}  // namespace

ThreadPoolCudart::~ThreadPoolCudart() {
  shutDown = true;
  cv.notify_all();
  for (auto &t : threadVector) {
    while (!t.joinable()) {
      cv.notify_all();
    }
    t.join();
  }
}

void ThreadPoolCudart::submit(TaskCudart &&task) {
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
}

ThreadPoolCudart::ThreadPoolCudart(const int localRank, const int threadNum,
                                   const std::vector<int> &bindingMap) {
  if (threadNum <= 0) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Wrong thread num");
  }
  if (!bindingMap.empty() &&
      bindingMap.size() != static_cast<std::size_t>(threadNum)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "bindingMap size incorrect");
  }
  threadVector.reserve(threadNum);
  for (int i = 0; i < threadNum; ++i) {
    threadVector.emplace_back(threadTask, localRank, &taskQueue, &queueMutex,
                              &cv, &cvMutex, &shutDown);
  }
  if (!bindingMap.empty()) {
    for (int i = 0; i < threadNum; ++i) {
      setThreadAffinity(threadVector[i], bindingMap[i]);
    }
  }
}
}  // namespace dllm
