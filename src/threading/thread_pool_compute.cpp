#include "threading/thread_pool_compute.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
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

void threadTask(const int localRank, std::queue<TaskCompute> *taskQueue,
                std::mutex *queueMutex, std::condition_variable *cv,
                std::mutex *cvMutex, std::atomic<bool> *shutDown) {
  ContextCompute context{.deviceRank = localRank};
  {
    int init = false;
    CHECK_MPI(MPI_Initialized(&init));
    int worldRank = 0;
    if (init) {
      CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    }
    context.curandSeed = worldRank;
    context.curandOffset = reinterpret_cast<unsigned long>(&context);
  }
  CHECK_CUDART(cudaSetDevice(localRank));
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  CHECK_CUDART(cudaDeviceGetDefaultMemPool(&context.memPool, localRank));
  uint64_t threshold = UINT64_MAX;
  CHECK_CUDART(cudaMemPoolSetAttribute(
      context.memPool, cudaMemPoolAttrReleaseThreshold, &threshold));
  int enable = 1;
  CHECK_CUDART(cudaMemPoolSetAttribute(
      context.memPool, cudaMemPoolReuseFollowEventDependencies, &enable));
  CHECK_CUDART(cudaMemPoolSetAttribute(
      context.memPool, cudaMemPoolReuseAllowOpportunistic, &enable));
  CHECK_CUDART(cudaMemPoolSetAttribute(
      context.memPool, cudaMemPoolReuseAllowInternalDependencies, &enable));
  CHECK_CUBLAS(cublasCreate_v2(&context.cublasHandle));
  CHECK_CUBLAS(cublasSetStream_v2(context.cublasHandle, context.cudaStream));
  while (!shutDown->load()) {
    TaskCompute task;
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
  CHECK_CUBLAS(cublasDestroy_v2(context.cublasHandle));
  CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
}

void threadTaskLight(const int localRank, std::queue<TaskCompute> *taskQueue,
                     std::mutex *queueMutex, std::condition_variable *cv,
                     std::mutex *cvMutex, std::atomic<bool> *shutDown) {
  ContextCompute context{.deviceRank = localRank};
  CHECK_CUDART(cudaSetDevice(localRank));
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  while (!shutDown->load()) {
    TaskCompute task;
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

ThreadPoolCompute::~ThreadPoolCompute() {
  shutDown = true;
  cv.notify_all();
  for (auto &t : threadVector) {
    while (!t.joinable()) {
      cv.notify_all();
    }
    t.join();
  }
}

std::shared_ptr<FutureCompute> ThreadPoolCompute::submit(TaskCompute &&task) {
  auto future = task.get_future();
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
  return std::make_shared<FutureCompute>(std::move(future));
}

ThreadPoolCompute::ThreadPoolCompute(int localRank, int threadNum,
                                     const std::vector<int> &bindingMap) {
  if (!bindingMap.empty() && bindingMap.size() != threadNum) {
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
