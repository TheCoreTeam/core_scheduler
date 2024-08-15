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
struct AssistQueue {
  int count;
  Event event;
  std::vector<at::Tensor> protected_tensors;
};

struct Impl_ final : Scheduler::Impl {
  explicit Impl_(int local_rank);

  void submit(Task &&task) override;

 private:
  std::vector<c10::cuda::CUDAStream> streams_;
  Event main_event_;
  std::deque<AssistQueue> assist_queue_;
  std::deque<AssistQueue> comm_queue_;
};

constexpr int assist_task_gap = 2;
constexpr int comm_task_gap = 5;
}  // namespace

Impl_::Impl_(const int local_rank) : Impl{local_rank} {
  streams_.reserve(Task::Impl::Priority::kNumPriority);
  for (int i = 0; i < Task::Impl::Priority::kNumPriority; ++i) {
    streams_.emplace_back(c10::cuda::getStreamFromPool(
        false, static_cast<c10::DeviceIndex>(local_rank)));
  }
}

void Impl_::submit(Task &&task) {
  c10::cuda::CUDAGuard deviceGuard{static_cast<c10::DeviceIndex>(device_rank_)};

  if (task.impl()->type() == Task::Impl::kConfig) {
    CS_NVTX_RANGE_FN(task.impl()->name());
    task();
    return;
  }

  auto call = [&](const Task::Impl::Priority tag) {
    for (auto &input = task.impl()->input(); auto &t : input) {
      if (t.impl()->priority() != tag) {
        if (!t.impl()->event().query()) {
          t.impl()->event().block();
        }
      }
    }
    CS_NVTX_RANGE_FN(task.impl()->name());
    task();
    for (auto &output = task.impl()->output(); auto &t : output) {
      t.impl()->priority() = tag;
      t.impl()->event().record();
    }
  };

  auto collect = [&]() {
    std::vector<at::Tensor> protected_tensors;
    protected_tensors.reserve(task.impl()->input().size() +
                              task.impl()->output().size() +
                              task.impl()->intermediate().size());
    for (const auto &t : task.impl()->input()) {
      protected_tensors.push_back(t.impl()->tensor());
    }
    for (const auto &t : task.impl()->output()) {
      protected_tensors.push_back(t.impl()->tensor());
    }
    for (const auto &t : task.impl()->intermediate()) {
      protected_tensors.push_back(t);
    }
    return protected_tensors;
  };

  if (task.impl()->priority() == Task::Impl::Priority::kMain) {
    c10::cuda::CUDAStreamGuard streamGuard{
        streams_[Task::Impl::Priority::kMain]};
    try {
      auto wait = [](std::deque<AssistQueue> &queue) {
        if (!queue.empty()) {
          for (auto &element : queue) {
            --element.count;
          }
          while (!queue.empty() && queue.front().count <= 0) {
            queue.front().event.block();
            queue.pop_front();
          }
        }
      };
      wait(assist_queue_);
      wait(comm_queue_);
      call(Task::Impl::Priority::kMain);
      main_event_.record();
    } catch (const std::exception &e) {
      CS_ASSERT_TRUE(false, fmt::format("Task {} failed with error: {}",
                                        task.impl()->name(), e.what()));
    }
  } else if (task.impl()->priority() == Task::Impl::Priority::kAssist) {
    c10::cuda::CUDAStreamGuard streamGuard{
        streams_[Task::Impl::Priority::kAssist]};
    main_event_.block();
    call(Task::Impl::Priority::kAssist);
    auto protected_tensors = collect();
    Event event;
    event.record();
    assist_queue_.emplace_front(assist_task_gap, std::move(event),
                                std::move(protected_tensors));
  } else {
    c10::cuda::CUDAStreamGuard streamGuard{
        streams_[Task::Impl::Priority::kComm]};
    main_event_.block();
    call(Task::Impl::Priority::kComm);
    auto protected_tensors = collect();
    Event event;
    event.record();
    comm_queue_.emplace_front(comm_task_gap, std::move(event),
                              std::move(protected_tensors));
  }
}

DynamicScheduler::DynamicScheduler(int localRank) {
  impl_ = std::make_shared<Impl_>(localRank);
}
}  // namespace cs
