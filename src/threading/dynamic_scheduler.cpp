/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "threading/dynamic_scheduler.h"

#include <c10/cuda/CUDAGuard.h>
#include <hwloc.h>
#include <hwloc/nvml.h>

#include <barrier>
#include <queue>
#include <unordered_set>

#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/event.h"
#include "threading/event_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs {
namespace {
hwloc_topology_t getHwlocTopology() {
  static struct Topo {
    hwloc_topology_t topo{};

    Topo() {
      nvmlInit();
      hwloc_topology_init(&topo);
      hwloc_topology_load(topo);
    }

    ~Topo() {
      hwloc_topology_destroy(topo);
      nvmlShutdown();
    }
  } topo;
  return topo.topo;
}

template <typename Fn>
std::shared_ptr<hwloc_cpuset_t> getBitmap(Fn &&fn) {
  return std::shared_ptr<hwloc_cpuset_t>{fn(new hwloc_cpuset_t),
                                         [](const hwloc_cpuset_t *bitmap) {
                                           hwloc_bitmap_free(*bitmap);
                                           delete bitmap;
                                         }};
}

hwloc_cpuset_t getCurrentCpuSet() {
  static auto guard = getBitmap([](hwloc_cpuset_t *bitmap) {
    *bitmap = hwloc_bitmap_alloc();
    const int num_cpus =
        hwloc_get_nbobjs_by_type(getHwlocTopology(), HWLOC_OBJ_PU);
    for (int i = 0; i < num_cpus; ++i) {
      const hwloc_obj_t obj =
          hwloc_get_obj_by_type(getHwlocTopology(), HWLOC_OBJ_PU, i);
      hwloc_bitmap_set(*bitmap, obj->os_index);
    }
    return bitmap;
  });
  return *guard;
}

void setClosestCpuSetToGpu(const hwloc_cpuset_t bitmap,
                           const unsigned int gpuRank) {
  nvmlDevice_t nvml_device;
  nvmlDeviceGetHandleByIndex(gpuRank, &nvml_device);
  CS_ASSERT_TRUE(hwloc_nvml_get_device_cpuset(getHwlocTopology(), nvml_device,
                                              bitmap) == 0,
                 "device rank not found by NVML");
}

hwloc_cpuset_t getClosestCpuSetToGpu(const unsigned int gpuRank) {
  static std::unordered_map<unsigned int, std::shared_ptr<hwloc_cpuset_t>> set;
  if (const auto find = set.find(gpuRank); find == set.end()) {
    auto bitmap = getBitmap([](hwloc_cpuset_t *bitmap) { return bitmap; });
    setClosestCpuSetToGpu(*bitmap, gpuRank);
    set.insert({gpuRank, bitmap});
    return *bitmap;
  } else {
    return *find->second;
  }
}
}  // namespace
namespace {
struct Impl_ final : Scheduler::Impl {
  explicit Impl_(int localRank);

  ~Impl_() override;

  void submit(Task &&task) override;

  void submit(const Task &task) = delete;

 private:
  int8_t threadCommIdx_;
  std::vector<std::shared_ptr<std::barrier<>>> startBarrier_{};
  std::vector<std::shared_ptr<std::barrier<>>> endBarrier_{};
  std::vector<std::jthread> threadVector_{};
  std::vector<std::shared_ptr<std::queue<Task>>> taskQueue_{};
  std::vector<std::vector<void *>> lastOutput_;
  std::shared_ptr<std::atomic<bool>> shutDown_{
      std::make_shared<std::atomic<bool>>(false)};
};

struct EventVectorPair {
  Event event{nullptr};
  std::vector<at::Tensor> tensors;
};

void memoryWatchDog(const std::shared_ptr<std::atomic<bool>> shutDown,
                    const int localRank,
                    std::shared_ptr<std::queue<EventVectorPair>> queue,
                    std::shared_ptr<std::mutex> queueMutex,
                    std::shared_ptr<std::condition_variable> cv,
                    std::shared_ptr<std::mutex> cvMutex) {
  c10::cuda::CUDAGuard deviceGuard{static_cast<c10::DeviceIndex>(localRank)};
  while (!shutDown->load()) {
    EventVectorPair pair;
    std::unique_lock lock{*queueMutex};
    if (!queue->empty()) {
      pair = std::move(queue->front());
      queue->pop();
    }
    lock.unlock();
    if (!pair.tensors.empty()) {
      try {
        pair.event.synchronize();
        pair.tensors.clear();
        pair.event = Event{nullptr};
      } catch (const std::exception &e) {
        CS_ASSERT_TRUE(false, "Task failed with error: {}", e.what());
      }
    } else {
      std::unique_lock uniqueLock{*cvMutex};
      cv->wait(uniqueLock, [&] { return shutDown->load() || !queue->empty(); });
    }
  }
}

void threadTask(const int localRank, const int8_t streamIdx,
                std::shared_ptr<std::queue<Task>> taskQueue,
                const std::shared_ptr<std::atomic<bool>> shutDown,
                const std::shared_ptr<std::barrier<>> startBarrier,
                const std::shared_ptr<std::barrier<>> endBarrier) {
  struct ContextCompute {
    int deviceRank{0};
    cudaStream_t cudaStream{nullptr};
  };
  ContextCompute context{.deviceRank = localRank};
  startBarrier->arrive_and_wait();
  const auto stream = c10::cuda::getStreamFromPool(
      false, static_cast<c10::DeviceIndex>(context.deviceRank));
  context.cudaStream = stream.stream();
  c10::cuda::CUDAStreamGuard streamGuard{stream};
  c10::cuda::CUDAGuard deviceGuard{
      static_cast<c10::DeviceIndex>(context.deviceRank)};

  struct WatchDogMeta {
    std::shared_ptr<std::queue<EventVectorPair>> queue{
        std::make_shared<std::queue<EventVectorPair>>()};
    std::shared_ptr<std::mutex> queueMutex{std::make_shared<std::mutex>()};
    std::shared_ptr<std::condition_variable> cv{
        std::make_shared<std::condition_variable>()};
    std::shared_ptr<std::mutex> cvMutex{std::make_shared<std::mutex>()};
  } watchDogMeta;

  std::jthread watchDog{memoryWatchDog,
                        shutDown,
                        localRank,
                        watchDogMeta.queue,
                        watchDogMeta.queueMutex,
                        watchDogMeta.cv,
                        watchDogMeta.cvMutex};

  while (true) {
    Event event{nullptr};
    startBarrier->arrive_and_wait();
    if (shutDown->load()) {
      break;
    }
    auto task = std::move(taskQueue->front());
    taskQueue->pop();
    try {
      for (auto &input = task.input(); auto &t : input) {
        if (t.impl()->streamIdx() != streamIdx) {
          if (!t.impl()->event().query()) {
            t.impl()->event().block();
          }
        }
      }
      CS_NVTX_RANGE_FN(task.name());
      task();
      if (!task.impl()->intermediate().empty()) {
        event = Event{};
        event.record();
      }
      for (auto &output = task.output(); auto &t : output) {
        t.impl()->streamIdx() = streamIdx;
        t.impl()->event().record();
      }
    } catch (const std::exception &e) {
      CS_ASSERT_TRUE(false, fmt::format("Task {} failed with error: {}",
                                        task.name(), e.what()));
    }
    for (auto &output = task.output(); auto &t : output) {
      t.impl()->stream() = context.cudaStream;
    }
    endBarrier->arrive_and_wait();
    if (!task.impl()->intermediate().empty()) {
      std::lock_guard guard{*watchDogMeta.queueMutex};
      watchDogMeta.queue->emplace(std::move(event),
                                  std::move(task.impl()->intermediate()));
      watchDogMeta.cv->notify_one();
    }
    task.reset();
  }

  watchDogMeta.cv->notify_one();
}

void threadCommTask(const int localRank, const int8_t streamIdx,
                    std::shared_ptr<std::queue<Task>> taskQueue,
                    const std::shared_ptr<std::atomic<bool>> shutDown,
                    const std::shared_ptr<std::barrier<>> startBarrier,
                    const std::shared_ptr<std::barrier<>> endBarrier) {
  struct ContextCompute {
    int deviceRank{0};
    cudaStream_t cudaStream{nullptr};
  };
  ContextCompute context{.deviceRank = localRank};
  startBarrier->arrive_and_wait();
  const auto stream = c10::cuda::getStreamFromPool(
      true, static_cast<c10::DeviceIndex>(context.deviceRank));
  context.cudaStream = stream.stream();
  c10::cuda::CUDAStreamGuard streamGuard{stream};
  c10::cuda::CUDAGuard deviceGuard{
      static_cast<c10::DeviceIndex>(context.deviceRank)};

  while (true) {
    startBarrier->arrive_and_wait();
    if (shutDown->load()) {
      break;
    }
    auto task = std::move(taskQueue->front());
    taskQueue->pop();
    try {
      for (auto &input = task.input(); auto &t : input) {
        if (t.impl()->streamIdx() != streamIdx) {
          if (!t.impl()->event().query()) {
            t.impl()->event().block();
          }
        }
      }
      CS_NVTX_RANGE_FN(task.name());
      task();
      for (auto &output = task.output(); auto &t : output) {
        t.impl()->streamIdx() = streamIdx;
        t.impl()->event().record();
      }
    } catch (const std::exception &e) {
      CS_ASSERT_TRUE(false, fmt::format("Task {} failed with error: {}",
                                        task.name(), e.what()));
    }
    for (auto &output = task.output(); auto &t : output) {
      t.impl()->stream() = context.cudaStream;
    }
    endBarrier->arrive_and_wait();
    task.reset();
  }
}
}  // namespace

Impl_::Impl_(int localRank) : Impl{localRank} {
  constexpr int threadNum = 2;
  taskQueue_.resize(threadNum);
  for (auto &queue : taskQueue_) {
    queue = std::make_shared<std::queue<Task>>();
  }
  lastOutput_.resize(threadNum);
  threadVector_.reserve(threadNum);
  startBarrier_.reserve(threadNum);
  endBarrier_.reserve(threadNum);
  for (int8_t i = 0; i < threadNum; ++i) {
    startBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
    endBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
    threadVector_.emplace_back(threadTask, localRank, i, taskQueue_[i],
                               shutDown_, startBarrier_[i], endBarrier_[i]);
    startBarrier_[i]->arrive_and_wait();
  }
  threadCommIdx_ = static_cast<int8_t>(threadVector_.size());
  taskQueue_.emplace_back(std::make_shared<std::queue<Task>>());
  startBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
  endBarrier_.emplace_back(std::make_shared<std::barrier<>>(2));
  lastOutput_.emplace_back();
  threadVector_.emplace_back(
      threadCommTask, localRank, threadCommIdx_, taskQueue_[threadCommIdx_],
      shutDown_, startBarrier_[threadCommIdx_], endBarrier_[threadCommIdx_]);
  startBarrier_[threadCommIdx_]->arrive_and_wait();
}

Impl_::~Impl_() {
  shutDown_->store(true);
  for (const auto &b : startBarrier_) {
    (void)b->arrive();
  }
  for (const auto &b : endBarrier_) {
    (void)b->arrive();
  }
}

void Impl_::submit(Task &&task) {
  int8_t streamIdx = -1;
  if (task.impl()->type() == Task::Impl::kNccl) {
    streamIdx = threadCommIdx_;
  } else {
    [&]() mutable {
      auto &input = task.input();
      std::unordered_set<void *> elements;
      for (const auto &ptr : input) {
        elements.insert(ptr.impl().get());
      }
      for (std::size_t i = 0; i < lastOutput_.size(); ++i) {
        for (auto e : lastOutput_[i]) {
          if (elements.contains(e)) {
            streamIdx = static_cast<int8_t>(i);
            return;
          }
        }
      }
    }();
    if (streamIdx == -1 /* not found */) {
      std::vector<int> count;
      count.resize(taskQueue_.size());
      std::ranges::fill(count, 0);
      for (auto &input = task.input(); auto &t : input) {
        ++count[t.impl()->streamIdx()];
      }
      int smallestCount = task.input().size();
      streamIdx = 0;
      for (std::size_t i = 0; i < count.size(); ++i) {
        if (count[i] < smallestCount) {
          smallestCount = count[i];
          streamIdx = i;
        }
      }
    }
  }

  taskQueue_[streamIdx]->push(task);
  (void)startBarrier_[streamIdx]->arrive();
  lastOutput_[streamIdx].clear();
  for (auto &output = task.output(); auto &t : output) {
    lastOutput_[streamIdx].push_back(t.impl().get());
  }
  endBarrier_[streamIdx]->arrive_and_wait();
}

DynamicScheduler::DynamicScheduler(int localRank) {
  impl_ = std::make_shared<Impl_>(localRank);
}
}  // namespace cs
