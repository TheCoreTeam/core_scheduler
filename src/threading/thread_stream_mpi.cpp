#include "threading/thread_stream_mpi.h"

#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>

#include "logger.h"

namespace dllm {
struct ThreadStreamMpi::Impl {
  explicit Impl(const ContextMpi &context, std::optional<const int> bindingMap);

  ~Impl();

  void submit(TaskMpi &&task);

  void submit(const TaskMpi &task) = delete;

  [[nodiscard]] int64_t commSize() const;

  [[nodiscard]] int64_t rank() const;

 private:
  const ContextMpi context_;
  std::latch latch_;
  std::queue<TaskMpi> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
  std::jthread thread{};
};

namespace {
void setThreadAffinity(std::jthread &th, const int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);

  int rc =
      pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
  DLLM_ASSERT_TRUE(rc == 0, "core binding error with code {}", rc);
}

void threadTask(const ContextMpi *context, std::queue<TaskMpi> *taskQueue,
                std::mutex *queueMutex, std::condition_variable *cv,
                std::mutex *cvMutex, const std::atomic<bool> *shutDown,
                std::latch *barrier) {
  barrier->arrive_and_wait();
  while (!shutDown->load()) {
    TaskMpi task;
    std::unique_lock lock{*queueMutex};
    if (!taskQueue->empty()) {
      task = std::move(taskQueue->front());
      taskQueue->pop();
    }
    lock.unlock();
    if (task.valid()) {
      try {
        task(context);
      } catch (const std::exception &e) {
        DLLM_ASSERT_TRUE(false, "Task failed with error: {}", e.what());
      }
    } else {
      std::unique_lock<std::mutex> uniqueLock{*cvMutex};
      cv->wait(uniqueLock,
               [&] { return shutDown->load() || !taskQueue->empty(); });
    }
  }
  std::unique_lock lock{*queueMutex};
  *taskQueue = {};
  lock.unlock();
}
}  // namespace

ThreadStreamMpi::Impl::Impl(const ContextMpi &context,
                            const std::optional<const int> bindingMap)
    : context_{context}, latch_{2}, thread{threadTask,  &context_, &taskQueue,
                                           &queueMutex, &cv,       &cvMutex,
                                           &shutDown,   &latch_} {
  if (bindingMap.has_value()) {
    setThreadAffinity(thread, bindingMap.value());
  }
  latch_.arrive_and_wait();
}

ThreadStreamMpi::Impl::~Impl() {
  shutDown = true;
  cv.notify_one();
  while (!thread.joinable()) {
    cv.notify_one();
  }
  thread.join();
}

void ThreadStreamMpi::Impl::submit(TaskMpi &&task) {
  std::unique_lock lock{queueMutex};
  taskQueue.push(std::move(task));
  lock.unlock();
  cv.notify_one();
}

int64_t ThreadStreamMpi::Impl::commSize() const { return context_.commSize; }

int64_t ThreadStreamMpi::Impl::rank() const { return context_.mpiRank; }

void ThreadStreamMpi::submit(TaskMpi &&task) const {
  impl_->submit(std::move(task));
}

int64_t ThreadStreamMpi::commSize() const { return impl_->commSize(); }

int64_t ThreadStreamMpi::rank() const { return impl_->rank(); }

ThreadStreamMpi::ThreadStreamMpi(const ContextMpi &context,
                                 const std::optional<const int> bindingMap)
    : impl_{std::make_shared<Impl>(context, bindingMap)} {}
}  // namespace dllm
