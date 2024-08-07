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

#pragma once
#include <mpi.h>

#include <memory>

#include "threading/scheduler.h"

namespace cs::communication {
struct Comm {
  struct Impl;

  explicit Comm(std::shared_ptr<Impl> impl);

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

  CS_API [[nodiscard]] int64_t get_rank() const;

  CS_API [[nodiscard]] int64_t get_size() const;

 private:
  std::shared_ptr<Impl> impl_;
};

struct Bucket {
  struct Impl;

  CS_API void apply(const Scheduler &scheduler, const Comm &comm) const;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

 protected:
  std::shared_ptr<Impl> impl_;
};

enum BackendType { kMPI, kNCCL, kNVP2P };

enum Operation { kSUM, kAVG };

CS_API Comm get_comm(MPI_Comm group, BackendType backendType);

CS_API Comm get_comm_world(BackendType backendType);

CS_API Comm get_comm_node(BackendType backendType);
}  // namespace cs::communication
