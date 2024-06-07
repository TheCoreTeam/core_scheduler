#include "threading/thread_stream_cudart.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>

#include <latch>
#include <queue>

#include "logger.h"
#include "threading/scheduler_impl.h"

namespace dllm {
namespace {
struct Impl_ final : Scheduler::Impl {
  explicit Impl_(int deviceRank, std::optional<const int> bindingMap);

  ~Impl_() override;

  void submit(TaskCudart &&task) override;

  void submit(const TaskCudart &task) = delete;

 private:
  std::latch latch_;
  std::queue<TaskCudart> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
  std::jthread thread{};
};

void setThreadAffinity(std::jthread &th, const int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);

  int rc =
      pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
  DLLM_ASSERT_TRUE(rc == 0, "core binding error with code {}", rc);
}

void threadTask(const int deviceRank, std::queue<TaskCudart> *taskQueue,
                std::mutex *queueMutex, std::condition_variable *cv,
                std::mutex *cvMutex, const std::atomic<bool> *shutDown,
                std::latch *barrier) {
  ContextCudart context;
  barrier->arrive_and_wait();
  CHECK_CUDART(cudaSetDevice(deviceRank));
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  const auto stream = c10::cuda::getStreamFromExternal(
      context.cudaStream, static_cast<c10::DeviceIndex>(context.deviceRank));
  c10::cuda::CUDAStreamGuard streamGuard{stream};
  c10::cuda::CUDAGuard deviceGuard{
      static_cast<c10::DeviceIndex>(context.deviceRank)};

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
        DLLM_ASSERT_TRUE(false, "Task failed with error: {}", e.what());
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

Impl_::Impl_(const int deviceRank, const std::optional<const int> bindingMap)
    : latch_{2}, thread{threadTask, deviceRank, &taskQueue, &queueMutex,
                        &cv,        &cvMutex,   &shutDown,  &latch_} {
  if (bindingMap.has_value()) {
    setThreadAffinity(thread, bindingMap.value());
  }
  latch_.arrive_and_wait();
}

Impl_::~Impl_() {
  shutDown = true;
  cv.notify_one();
  while (!thread.joinable()) {
    cv.notify_one();
  }
  thread.join();
}

void Impl_::submit(TaskCudart &&task) {
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
}

void ThreadStreamCudart::submit(TaskCudart &&task) const {
  impl_->submit(std::move(task));
}

ThreadStreamCudart::ThreadStreamCudart(
    const int deviceRank, const std::optional<const int> bindingMap) {
  impl_ = std::make_shared<Impl_>(deviceRank, bindingMap);
}
}  // namespace dllm
