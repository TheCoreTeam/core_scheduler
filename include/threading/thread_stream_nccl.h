#pragma once

#include <optional>

#include "threading/context_nccl.h"
#include "threading/scheduler.h"

namespace dllm {
struct ThreadStreamNccl : Scheduler {
  ThreadStreamNccl(MPI_Comm mpiComm, int deviceRank,
                   std::optional<const int> bindingMap = {});

  [[nodiscard]] int64_t commSize() const;

  [[nodiscard]] int64_t rank() const;
};
}  // namespace dllm
