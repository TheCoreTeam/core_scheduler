#pragma once
#include <cuda_runtime.h>
#include <mpi.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace dllm {
struct ContextNccl {
  MPI_Comm mpiComm;
  int mpiRank;
  int commSize;
  at::intrusive_ptr<c10d::Store> store;
  std::unique_ptr<c10d::Backend> backend;
  cudaStream_t cudaStream;
};
}  // namespace dllm