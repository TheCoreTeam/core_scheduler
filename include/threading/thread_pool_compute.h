#pragma once
#include <vector>

#include "threading/scheduler.h"

namespace dllm {
// This is only for anything, including computation (cublas, cublasLt, etc.) as
// well as communication (MPI, NCCL, etc.). However, this thread pool requires
// more resource (e.g., GPU context memory).
struct ThreadPoolCompute : Scheduler {
  ThreadPoolCompute(int localRank, int threadNum,
                    const std::vector<int> &bindingMap = {});
};
}  // namespace dllm
