#include <gtest/gtest.h>
#include <mpi.h>

#include <Eigen/Dense>

#include "communication/mpi/all_gather.h"
#include "communication/nccl/all_gather.h"
#include "dtensor_mpi.h"
#include "dtensor_nccl.h"
#include "logger.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_mpi.h"
#include "threading/thread_stream_nccl.h"
#include "util.h"

class AllGatherMPITestFixture : public ::testing::Test {
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
};

namespace {
template <typename T>
void TestAllGatherT(const dllm::ContextMpi &contextMpi) {
  int commSize;
  CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &commSize));
  const int blockSize = 128;
  const dllm::TensorIndexType m = blockSize * commSize;
  const auto shapeX = cute::make_shape(m);
  const auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

  Eigen::Vector<T, Eigen::Dynamic> x(m);

  auto tensorX = std::make_shared<dllm::Tensor1D>(
      dllm::Tensor1D{x.data(), layoutX, dllm::toDtype<T>(), dllm::CPU});

  std::srand(contextMpi.mpiRank + 1);
  x.setRandom();

  auto task =
      dllm::communication::AllGather<dllm::communication::MPI>::runInplace(
          tensorX);
  tensorX.reset();
  task(&contextMpi);

  Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
  if (contextMpi.mpiRank == 0) {
    int worldSize;
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
    accumulator.setZero();
    for (int i = 0; i < worldSize; ++i) {
      std::srand(i + 1);
      accumulator.segment(i * blockSize, blockSize) =
          Eigen::Vector<T, Eigen::Dynamic>(m).setRandom().segment(i * blockSize,
                                                                  blockSize);
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
}
}  // namespace

TEST_F(AllGatherMPITestFixture, TestForwardF32) {
  TestAllGatherT<float>(contextMpi);
}
TEST_F(AllGatherMPITestFixture, TestForwardF64) {
  TestAllGatherT<double>(contextMpi);
}

class AllGatherMPIThreadPoolComputeTestFixture : public ::testing::Test {
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
  dllm::ThreadStreamMpi threadStreamMpi{contextMpi};
};

namespace {
template <typename T>
void TestThreadPoolComputeAllGatherT(dllm::ThreadStreamMpi &threadStreamMpi,
                                     dllm::ContextMpi &contextMpi) {
  int commSize;
  CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &commSize));
  const int blockSize = 128;
  const dllm::TensorIndexType m = blockSize * commSize;
  const auto shapeX = cute::make_shape(m);
  const auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

  Eigen::Vector<T, Eigen::Dynamic> x(m);

  auto tensorX = std::make_shared<dllm::Tensor1D>(
      dllm::Tensor1D{x.data(), layoutX, dllm::toDtype<T>(), dllm::CUDA});

  std::srand(contextMpi.mpiRank + 1);
  x.setRandom();

  auto task =
      dllm::communication::AllGather<dllm::communication::MPI>::runInplace(
          tensorX);
  threadStreamMpi.submit(std::move(task));
  {
    dllm::util::FutureGuard{tensorX->future->rFuture};
    dllm::util::FutureGuard{tensorX->future->wFuture};
  }

  Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
  if (contextMpi.mpiRank == 0) {
    int worldSize;
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
    accumulator.setZero();
    for (int i = 0; i < worldSize; ++i) {
      std::srand(i + 1);
      accumulator.segment(i * blockSize, blockSize) =
          Eigen::Vector<T, Eigen::Dynamic>(m).setRandom().segment(i * blockSize,
                                                                  blockSize);
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
}
}  // namespace

TEST_F(AllGatherMPIThreadPoolComputeTestFixture, TestForwardF32) {
  TestThreadPoolComputeAllGatherT<float>(threadStreamMpi, contextMpi);
}
TEST_F(AllGatherMPIThreadPoolComputeTestFixture, TestForwardF64) {
  TestThreadPoolComputeAllGatherT<double>(threadStreamMpi, contextMpi);
}

class AllGatherNcclTestFixture : public ::testing::Test {
 protected:
  int rank;
  dllm::ContextMpi contextMpi;
  dllm::ContextNccl contextNccl;

  void SetUp() override {
    CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &contextMpi.mpiComm));
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &contextMpi.commSize));
    CHECK_MPI(MPI_Comm_rank(contextMpi.mpiComm, &contextMpi.mpiRank));
    ncclUniqueId id;
    if (contextMpi.mpiRank == 0) {
      CHECK_NCCL(ncclGetUniqueId(&id));
    }
    CHECK_MPI(
        MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, contextMpi.mpiComm));
    contextNccl.ncclRank = contextMpi.mpiRank;
    contextNccl.commSize = contextMpi.commSize;
    CHECK_CUDART(cudaSetDevice(contextMpi.mpiRank));
    CHECK_NCCL(ncclCommInitRank(&contextNccl.ncclComm, contextMpi.commSize, id,
                                contextMpi.mpiRank));
  }

  void TearDown() override {
    CHECK_NCCL(ncclCommDestroy(contextNccl.ncclComm));
    CHECK_MPI(MPI_Comm_free(&contextMpi.mpiComm));
  }
};

namespace {
template <typename T>
void TestNcclAllGatherT(const dllm::ContextMpi &contextMpi,
                        const dllm::ContextNccl &contextNccl) {
  int commSize;
  CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &commSize));
  const int blockSize = 128;
  const dllm::TensorIndexType m = blockSize * commSize;
  const auto shapeX = cute::make_shape(m);
  const auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

  Eigen::Vector<T, Eigen::Dynamic> x(m);
  std::srand(contextMpi.mpiRank + 1);
  x.setRandom();

  T *xDev;
  CHECK_CUDART(cudaMalloc(&xDev, sizeof(T) * cute::size(layoutX)));
  CHECK_CUDART(cudaMemcpy(xDev, x.data(), sizeof(T) * cute::size(layoutX),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorX = std::make_shared<dllm::Tensor1D>(
      dllm::Tensor1D{xDev, layoutX, dllm::toDtype<T>(), dllm::CUDA});

  auto task =
      dllm::communication::AllGather<dllm::communication::NCCL>::runInplace(
          tensorX);
  task(&contextNccl);
  {
    dllm::util::FutureGuard{tensorX->future->rFuture};
    dllm::util::FutureGuard{tensorX->future->wFuture};
  }

  CHECK_CUDART(cudaMemcpy(x.data(), xDev, sizeof(T) * cute::size(layoutX),
                          cudaMemcpyDeviceToHost));

  Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
  if (contextMpi.mpiRank == 0) {
    int worldSize;
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
    accumulator.setZero();
    for (int i = 0; i < worldSize; ++i) {
      std::srand(i + 1);
      accumulator.segment(i * blockSize, blockSize) =
          Eigen::Vector<T, Eigen::Dynamic>(m).setRandom().segment(i * blockSize,
                                                                  blockSize);
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
  CHECK_CUDART(cudaFree(xDev));
}
}  // namespace

TEST_F(AllGatherNcclTestFixture, TestForwardF32) {
  TestNcclAllGatherT<float>(contextMpi, contextNccl);
}
TEST_F(AllGatherNcclTestFixture, TestForwardF64) {
  TestNcclAllGatherT<double>(contextMpi, contextNccl);
}

class AllGatherThreadStreamNcclTestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};
  int rank;
  dllm::ContextMpi contextMpi;
  dllm::ThreadStreamNccl *threadStreamNccl;

  AllGatherThreadStreamNcclTestFixture() {
    int processesPerNode;
    CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &contextMpi.mpiComm));
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &processesPerNode));
    CHECK_MPI(MPI_Comm_rank(contextMpi.mpiComm, &contextMpi.mpiRank));
    ncclUniqueId id;
    if (contextMpi.mpiRank == 0) {
      CHECK_NCCL(ncclGetUniqueId(&id));
    }
    CHECK_MPI(
        MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, contextMpi.mpiComm));
    threadStreamNccl = new dllm::ThreadStreamNccl{
        id, processesPerNode, contextMpi.mpiRank, contextMpi.mpiRank};
    CHECK_CUDART(cudaSetDevice(contextMpi.mpiRank));
  }

  ~AllGatherThreadStreamNcclTestFixture() {
    delete threadStreamNccl;
    CHECK_MPI(MPI_Comm_free(&contextMpi.mpiComm));
  }
};

namespace {
template <typename T>
void TestThreadStreamNcclAllGatherT(dllm::ThreadStreamNccl &threadStreamNccl,
                                    dllm::ContextMpi &contextMpi) {
  int commSize;
  CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &commSize));
  const int blockSize = 128;
  const dllm::TensorIndexType m = blockSize * commSize;
  const auto shapeX = cute::make_shape(m);
  const auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

  Eigen::Vector<T, Eigen::Dynamic> x(m);
  std::srand(contextMpi.mpiRank + 1);
  x.setRandom();

  T *xDev;
  CHECK_CUDART(cudaMalloc(&xDev, sizeof(T) * cute::size(layoutX)));
  CHECK_CUDART(cudaMemcpy(xDev, x.data(), sizeof(T) * cute::size(layoutX),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorX = std::make_shared<dllm::Tensor1D>(
      dllm::Tensor1D{xDev, layoutX, dllm::toDtype<T>(), dllm::CUDA});

  auto task =
      dllm::communication::AllGather<dllm::communication::NCCL>::runInplace(
          tensorX);
  threadStreamNccl.submit(std::move(task));
  {
    dllm::util::FutureGuard{tensorX->future->rFuture};
    dllm::util::FutureGuard{tensorX->future->wFuture};
  }
  CHECK_CUDART(cudaMemcpy(x.data(), xDev, sizeof(T) * cute::size(layoutX),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
  if (contextMpi.mpiRank == 0) {
    int worldSize;
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
    accumulator.setZero();
    for (int i = 0; i < worldSize; ++i) {
      std::srand(i + 1);
      accumulator.segment(i * blockSize, blockSize) =
          Eigen::Vector<T, Eigen::Dynamic>(m).setRandom().segment(i * blockSize,
                                                                  blockSize);
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
  CHECK_CUDART(cudaFree(xDev));
}
}  // namespace

TEST_F(AllGatherThreadStreamNcclTestFixture, TestForwardF32) {
  TestThreadStreamNcclAllGatherT<float>(*threadStreamNccl, contextMpi);
}
TEST_F(AllGatherThreadStreamNcclTestFixture, TestForwardF64) {
  TestThreadStreamNcclAllGatherT<double>(*threadStreamNccl, contextMpi);
}