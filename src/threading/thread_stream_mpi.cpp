#include "threading/thread_stream_mpi.h"

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

void threadTask(const ContextMpi context, std::queue<TaskMpi> *taskQueue,
                std::mutex *queueMutex, std::condition_variable *cv,
                std::mutex *cvMutex, std::atomic<bool> *shutDown) {
  while (!shutDown->load()) {
    TaskMpi task;
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
}
}  // namespace

ThreadStreamMpi::~ThreadStreamMpi() {
  shutDown = true;
  cv.notify_one();
  if (!thread.joinable()) {
    cv.notify_all();
  }
  thread.join();
}

std::shared_ptr<FutureMpi> ThreadStreamMpi::submit(TaskMpi &&task) {
  auto future = task.get_future();
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
  return std::make_shared<FutureMpi>(std::move(future));
}

ThreadStreamMpi::ThreadStreamMpi(const ContextMpi context,
                                 const std::optional<const int> bindingMap)
    : thread{threadTask, context,  &taskQueue, &queueMutex,
             &cv,        &cvMutex, &shutDown} {
  if (bindingMap.has_value()) {
    setThreadAffinity(thread, bindingMap.value());
  }
}
}  // namespace dllm
