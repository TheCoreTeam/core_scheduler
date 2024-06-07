#include "threading/thread_pool_compute.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
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
  Impl_(int localRank, int threadNum, const std::vector<int> &bindingMap);

  ~Impl_() override;

  void submit(TaskCompute &&task) override;

  void submit(const TaskCompute &task) = delete;

 private:
  std::latch latch_;
  std::vector<std::jthread> threadVector{};
  std::queue<TaskCompute> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
};

void setThreadAffinity(std::jthread &th, const int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);

  int rc =
      pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
  DLLM_ASSERT_TRUE(rc == 0, "core binding error with code {}", rc);
}

void threadTask(const int localRank, std::queue<TaskCompute> *taskQueue,
                std::mutex *queueMutex, std::condition_variable *cv,
                std::mutex *cvMutex, const std::atomic<bool> *shutDown,
                std::latch *barrier) {
  ContextCompute context{.deviceRank = localRank};
  barrier->arrive_and_wait();
  CHECK_CUDART(cudaSetDevice(localRank));
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  CHECK_CUDART(cudaDeviceGetDefaultMemPool(&context.memPool, localRank));
  uint64_t threshold = UINT64_MAX;
  CHECK_CUDART(cudaMemPoolSetAttribute(
      context.memPool, cudaMemPoolAttrReleaseThreshold, &threshold));
  {
    int enable = 1;
    CHECK_CUDART(cudaMemPoolSetAttribute(
        context.memPool, cudaMemPoolReuseFollowEventDependencies, &enable));
    CHECK_CUDART(cudaMemPoolSetAttribute(
        context.memPool, cudaMemPoolReuseAllowOpportunistic, &enable));
    CHECK_CUDART(cudaMemPoolSetAttribute(
        context.memPool, cudaMemPoolReuseAllowInternalDependencies, &enable));
  }
  CHECK_CUBLAS(cublasCreate_v2(&context.cublasHandle));
  CHECK_CUBLAS(cublasSetStream_v2(context.cublasHandle, context.cudaStream));
  const auto stream = c10::cuda::getStreamFromExternal(
      context.cudaStream, static_cast<c10::DeviceIndex>(context.deviceRank));
  c10::cuda::CUDAStreamGuard streamGuard{stream};
  c10::cuda::CUDAGuard deviceGuard{
      static_cast<c10::DeviceIndex>(context.deviceRank)};

  while (!shutDown->load()) {
    TaskCompute task;
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
  CHECK_CUBLAS(cublasDestroy_v2(context.cublasHandle));
  CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
}
}  // namespace

Impl_::Impl_(int localRank, int threadNum, const std::vector<int> &bindingMap)
    : latch_{threadNum + 1} {
  DLLM_ASSERT_TRUE(threadNum > 0, "Wrong thread num");
  DLLM_ASSERT_TRUE(bindingMap.empty() ||
                       bindingMap.size() == static_cast<std::size_t>(threadNum),
                   "bindingMap size mismathches with threadNum");
  threadVector.reserve(threadNum);
  for (int i = 0; i < threadNum; ++i) {
    threadVector.emplace_back(threadTask, localRank, &taskQueue, &queueMutex,
                              &cv, &cvMutex, &shutDown, &latch_);
  }
  if (!bindingMap.empty()) {
    for (int i = 0; i < threadNum; ++i) {
      setThreadAffinity(threadVector[i], bindingMap[i]);
    }
  }
  latch_.arrive_and_wait();
}

Impl_::~Impl_() {
  shutDown = true;
  cv.notify_all();
  for (auto &t : threadVector) {
    while (!t.joinable()) {
      cv.notify_all();
    }
    t.join();
  }
}

void Impl_::submit(TaskCompute &&task) {
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
}

void ThreadPoolCompute::submit(TaskCompute &&task) const {
  impl_->submit(std::move(task));
}

ThreadPoolCompute::ThreadPoolCompute(const int localRank, const int threadNum,
                                     const std::vector<int> &bindingMap) {
  impl_ = std::make_shared<Impl_>(localRank, threadNum, bindingMap);
}
}  // namespace dllm
