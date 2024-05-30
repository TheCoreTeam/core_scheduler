#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/all_reduce.h"
#include "compute/utils.h"
#include "logger.h"
#include "memory/to_torch.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_cudart.h"
#include "threading/thread_stream_mpi.h"
#include "threading/thread_stream_nccl.h"

template <typename T>
struct TypeToTorch;

template <>
struct TypeToTorch<float> {
  using Type = float;
  static const at::ScalarType type = at::kFloat;
};

template <>
struct TypeToTorch<nv_half> {
  using Type = c10::Half;
  static const at::ScalarType type = at::kHalf;
};

template <>
struct TypeToTorch<double> {
  using Type = double;
  static const at::ScalarType type = at::kDouble;
};

class AllReduceMPITestFixture : public ::testing::Test {
 protected:
  dllm::ContextMpi contextMpi{
      [] {
        int rank;
        CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        return rank;
      }(),
      [] {
        int commSize;
        CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
        return commSize;
      }(),
      MPI_COMM_WORLD};

  dllm::ThreadStreamMpi stream{contextMpi};
  dllm::ThreadStreamCudart copy{0};
  dllm::ThreadPoolCompute tp{0, 3};

  template <typename T>
  void TestAllReduceT(int m);
};

template <typename T>
void AllReduceMPITestFixture::TestAllReduceT(const int m) {
  const at::Device device = at::kCPU;
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  at::manual_seed(stream.rank() + 1);
  auto x = dllm::Tensor::create();
  {
    auto task = dllm::compute::Utils::rand(x, {m}, option);
    tp.submit(std::move(task));
  }
  {
    auto task =
        dllm::communication::AllReduce<dllm::communication::MPI>::runInplace(
            x, dllm::communication::SUM);
    stream.submit(std::move(task));
  }

  at::Tensor x_torch;
  {
    auto task = dllm::memory::toTorch(x_torch, x);
    copy.submit(std::move(task));
    x->wait();
  }

  auto accumulator = torch::zeros_like(x_torch);
  if (contextMpi.mpiRank == 0) {
    for (int i = 0; i < stream.commSize(); ++i) {
      at::manual_seed(i + 1);
      accumulator += torch::rand({m}, option);
    }
    ASSERT_TRUE(at::allclose(x, x_torch));
  }
}

TEST_F(AllReduceMPITestFixture, TestForwardF32) { TestAllReduceT<float>(128); }
TEST_F(AllReduceMPITestFixture, TestForwardF64) { TestAllReduceT<double>(128); }

class AllReduceNcclTestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};
  dllm::ContextMpi contextMpi;
  dllm::ThreadStreamNccl *stream;
  dllm::ThreadStreamCudart *copy;
  dllm::ThreadPoolCompute *tp;

  AllReduceNcclTestFixture() {
    int processesPerNode;
    CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &contextMpi.mpiComm));
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &processesPerNode));
    CHECK_MPI(MPI_Comm_rank(contextMpi.mpiComm, &contextMpi.mpiRank));
    ncclUniqueId id;
    if (contextMpi.mpiRank == 0) {
      CHECK_NCCL(ncclGetUniqueId(&id));
    }
    CHECK_MPI(
        MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, contextMpi.mpiComm));
    stream = new dllm::ThreadStreamNccl{id, processesPerNode,
                                        contextMpi.mpiRank, contextMpi.mpiRank};
    copy = new dllm::ThreadStreamCudart{contextMpi.mpiRank};
    tp = new dllm::ThreadPoolCompute{contextMpi.mpiRank, 3};
    CHECK_CUDART(cudaSetDevice(contextMpi.mpiRank));
  }

  ~AllReduceNcclTestFixture() {
    delete tp;
    delete copy;
    delete stream;
    CHECK_MPI(MPI_Comm_free(&contextMpi.mpiComm));
  }

  template <typename T>
  void TestAllReduceT(int m);
};

template <typename T>
void AllReduceNcclTestFixture::TestAllReduceT(const int m) {
  const at::Device device(at::kCUDA, stream->rank());
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  at::manual_seed(stream->rank() + 1);
  auto x = dllm::Tensor::create();
  {
    auto task = dllm::compute::Utils::rand(x, {m}, option);
    tp->submit(std::move(task));
  }
  {
    auto task =
        dllm::communication::AllReduce<dllm::communication::NCCL>::runInplace(
            x, dllm::communication::SUM);
    stream->submit(std::move(task));
  }

  at::Tensor x_torch;
  {
    auto task = dllm::memory::toTorch(x_torch, x);
    copy->submit(std::move(task));
    x->wait();
  }

  auto accumulator = torch::empty_like(x_torch);
  if (contextMpi.mpiRank == 0) {
    accumulator.zero_();
    for (int i = 0; i < stream->commSize(); ++i) {
      at::manual_seed(i + 1);
      accumulator += torch::rand({m}, option);
    }
    GTEST_ASSERT_TRUE(at::allclose(x, x_torch));
  }
}

TEST_F(AllReduceNcclTestFixture, TestForwardF32) { TestAllReduceT<float>(128); }
TEST_F(AllReduceNcclTestFixture, TestForwardF64) {
  TestAllReduceT<double>(128);
}
