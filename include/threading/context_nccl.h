#pragma once
#include <cuda_runtime.h>
#include <nccl.h>

namespace dllm {
struct ContextNccl {
  cudaStream_t cudaStream{nullptr};
  int ncclRank;
  int commSize;
  ncclComm_t ncclComm{nullptr};
};
}  // namespace dllm