#include "threading/thread_stream_cudart.h"

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

void threadTask(const int deviceRank, std::queue<TaskCudart> *taskQueue,
                std::mutex *queueMutex, std::condition_variable *cv,
                std::mutex *cvMutex, const std::atomic<bool> *shutDown) {
  ContextCudart context;
  CHECK_CUDART(cudaSetDevice(deviceRank));
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
      task(&context);
    } else {
      std::unique_lock<std::mutex> uniqueLock{*cvMutex};
      cv->wait(uniqueLock,
               [&] { return shutDown->load() || !taskQueue->empty(); });
    }
  }
  CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
}
}  // namespace

ThreadStreamCudart::~ThreadStreamCudart() {
  shutDown = true;
  cv.notify_one();
  while (!thread.joinable()) {
    cv.notify_one();
  }
  thread.join();
}

void ThreadStreamCudart::submit(TaskCudart &&task) {
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
}

ThreadStreamCudart::ThreadStreamCudart(
    const int deviceRank, const std::optional<const int> bindingMap)
    : thread{threadTask, deviceRank, &taskQueue, &queueMutex,
             &cv,        &cvMutex,   &shutDown} {
  if (bindingMap.has_value()) {
    setThreadAffinity(thread, bindingMap.value());
  }
}
}  // namespace dllm
