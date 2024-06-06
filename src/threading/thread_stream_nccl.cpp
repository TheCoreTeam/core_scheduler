#include "threading/thread_stream_nccl.h"

#include <arpa/inet.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <netdb.h>
#include <pthread.h>
#include <sched.h>

#include <barrier>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "logger.h"

namespace dllm {
namespace {
void setThreadAffinity(std::jthread &th, const int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);

  const int rc =
      pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
  DLLM_ASSERT_TRUE(rc == 0, "core binding error with code {}", rc);
}

void threadTask(const MPI_Comm mpiComm, const int deviceRank,
                std::queue<TaskNccl> *taskQueue, std::mutex *queueMutex,
                std::condition_variable *cv, std::mutex *cvMutex,
                const std::atomic<bool> *shutDown, std::latch *barrier) {
  barrier->arrive_and_wait();
  ContextNccl context;
  {
    int rank, size;
    MPI_Comm_rank(mpiComm, &rank);
    MPI_Comm_size(mpiComm, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    char addr0[INET_ADDRSTRLEN];  // INET_ADDRSTRLEN is typically 16 for IPv4

    if (rank == 0) {
      const hostent *he = gethostbyname(processor_name);
      DLLM_ASSERT_TRUE(
          he != nullptr,
          fmt::format("Error resolving hostname: {}", hstrerror(h_errno)));
      // Convert the first host address to a string
      strcpy(addr0, inet_ntoa(*reinterpret_cast<in_addr *>(he->h_addr)));
    }

    // Broadcast the IP address from rank 0 to all other ranks
    MPI_Bcast(addr0, INET_ADDRSTRLEN, MPI_CHAR, 0, mpiComm);

    context.mpiComm = mpiComm;
    context.mpiRank = rank;
    context.commSize = size;
    context.store = c10::make_intrusive<c10d::TCPStore>(
        addr0,
        c10d::TCPStoreOptions{.isServer = rank == 0, .numWorkers = size});
    context.backend =
        std::make_unique<c10d::ProcessGroupNCCL>(context.store, rank, size);
  }

  CHECK_CUDART(cudaSetDevice(deviceRank));
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
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

int64_t ThreadStreamNccl::commSize() const { return commSize_; }

int64_t ThreadStreamNccl::rank() const { return rank_; }

ThreadStreamNccl::ThreadStreamNccl(const MPI_Comm mpiComm, const int deviceRank,
                                   const std::optional<const int> bindingMap)
    : commSize_{[&] {
        int size;
        MPI_Comm_size(mpiComm, &size);
        return size;
      }()},
      rank_{[&] {
        int rank;
        MPI_Comm_rank(mpiComm, &rank);
        return rank;
      }()},
      latch_{2},
      thread{threadTask, mpiComm,  deviceRank, &taskQueue, &queueMutex,
             &cv,        &cvMutex, &shutDown,  &latch_} {
  if (bindingMap.has_value()) {
    setThreadAffinity(thread, bindingMap.value());
  }
  latch_.arrive_and_wait();
}
}  // namespace dllm
