#pragma once
#include <mpi.h>

#include <memory>

#include "threading/scheduler.h"

namespace dllm::communication {
struct Comm {
  struct Impl;

  explicit Comm(std::shared_ptr<Impl> impl) : impl_{std::move(impl)} {}

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

  [[nodiscard]] int64_t getRank() const;

  [[nodiscard]] int64_t getSize() const;

 private:
  std::shared_ptr<Impl> impl_;
};

struct Bucket {
  struct Impl;

  void apply(const Scheduler &scheduler, const Comm &comm) const;

  const std::shared_ptr<Impl> &impl() const;

 protected:
  std::shared_ptr<Impl> impl_;
};

enum BackendType { MPI, NCCL, NVP2P };

enum Operation { SUM };

Comm getComm(MPI_Comm group, BackendType backendType);

Comm getCommWorld(BackendType backendType);

Comm getCommNode(BackendType backendType);
}  // namespace dllm::communication
