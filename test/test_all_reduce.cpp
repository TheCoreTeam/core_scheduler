#include <gtest/gtest.h>
#include <mpi.h>

#include <Eigen/Dense>

#include "communication/mpi/all_reduce.h"
#include "communication/nccl/all_reduce.h"
#include "dtensor_mpi.h"
#include "dtensor_nccl.h"
#include "logger.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_mpi.h"
#include "threading/thread_stream_nccl.h"

class AllReduceMPITestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};
  int rank;
  dllm::ContextMpi contextMpi;

  void SetUp() override {
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &contextMpi.mpiRank));
    contextMpi.mpiComm = MPI_COMM_WORLD;
  }
};

namespace {
template <typename T>
void TestAllReduceT(const dllm::ContextCompute &context,
                    const dllm::ContextMpi &contextMpi) {
  const dllm::TensorIndexType m = 128;
  auto shapeX = cute::make_shape(m);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

  Eigen::Vector<T, Eigen::Dynamic> x(m);

  auto tensorX = std::make_shared<dllm::Tensor1D>(
      dllm::Tensor1D{x.data(), layoutX, dllm::toDtype<T>(), dllm::CUDA});

  std::srand(contextMpi.mpiRank + 1);
  x.setRandom();

  auto task =
      dllm::communication::AllReduce<dllm::communication::MPI>::runInplace(
          tensorX, dllm::communication::SUM);
  tensorX.reset();
  task(&contextMpi);

  Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
  if (contextMpi.mpiRank == 0) {
    int worldSize;
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
    accumulator.setZero();
    for (int i = 0; i < worldSize; ++i) {
      std::srand(i + 1);
      accumulator += Eigen::Vector<T, Eigen::Dynamic>(m).setRandom();
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
}
}  // namespace

TEST_F(AllReduceMPITestFixture, TestForwardF32) {
  TestAllReduceT<float>(context, contextMpi);
}
TEST_F(AllReduceMPITestFixture, TestForwardF64) {
  TestAllReduceT<double>(context, contextMpi);
}

class AllReduceMPIThreadPoolComputeTestFixture : public ::testing::Test {
 protected:
  dllm::ContextMpi contextMpi{[] {
                                int rank;
                                CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
                                return rank;
                              }(),
                              MPI_COMM_WORLD};
  dllm::ThreadStreamMpi threadStreamMpi{contextMpi};
};

namespace {
template <typename T>
void TestThreadPoolComputeAllReduceT(dllm::ThreadStreamMpi &threadStreamMpi,
                                     dllm::ContextMpi &contextMpi) {
  const dllm::TensorIndexType m = 128;
  auto shapeX = cute::make_shape(m);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

  Eigen::Vector<T, Eigen::Dynamic> x(m);

  auto tensorX = std::make_shared<dllm::Tensor1D>(
      dllm::Tensor1D{x.data(), layoutX, dllm::toDtype<T>(), dllm::CUDA});

  std::srand(contextMpi.mpiRank + 1);
  x.setRandom();

  auto task =
      dllm::communication::AllReduce<dllm::communication::MPI>::runInplace(
          tensorX, dllm::communication::SUM);
  threadStreamMpi.submit(std::move(task));
  tensorX->future->wait();

  Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
  if (contextMpi.mpiRank == 0) {
    int worldSize;
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
    accumulator.setZero();
    for (int i = 0; i < worldSize; ++i) {
      std::srand(i + 1);
      accumulator += Eigen::Vector<T, Eigen::Dynamic>(m).setRandom();
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
}
}  // namespace

TEST_F(AllReduceMPIThreadPoolComputeTestFixture, TestForwardF32) {
  TestThreadPoolComputeAllReduceT<float>(threadStreamMpi, contextMpi);
}
TEST_F(AllReduceMPIThreadPoolComputeTestFixture, TestForwardF64) {
  TestThreadPoolComputeAllReduceT<double>(threadStreamMpi, contextMpi);
}

class AllReduceNcclTestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};
  int rank;
  dllm::ContextMpi contextMpi;
  dllm::ContextNccl contextNccl;

  void SetUp() override {
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
    contextNccl.ncclRank = contextMpi.mpiRank;
    CHECK_CUDART(cudaSetDevice(contextMpi.mpiRank));
    CHECK_NCCL(ncclCommInitRank(&contextNccl.ncclComm, processesPerNode, id,
                                contextMpi.mpiRank));
  }

  void TearDown() override {
    CHECK_NCCL(ncclCommDestroy(contextNccl.ncclComm));
    CHECK_MPI(MPI_Comm_free(&contextMpi.mpiComm));
  }
};

namespace {
template <typename T>
void TestNcclAllReduceT(const dllm::ContextCompute &context,
                        const dllm::ContextMpi &contextMpi,
                        const dllm::ContextNccl &contextNccl) {
  const dllm::TensorIndexType m = 128;
  auto shapeX = cute::make_shape(m);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

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
      dllm::communication::AllReduce<dllm::communication::NCCL>::runInplace(
          tensorX, dllm::communication::SUM);
  task(&contextNccl);
  tensorX->future->wait();

  CHECK_CUDART(cudaMemcpy(x.data(), xDev, sizeof(T) * cute::size(layoutX),
                          cudaMemcpyDeviceToHost));

  Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
  if (contextMpi.mpiRank == 0) {
    int worldSize;
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
    accumulator.setZero();
    for (int i = 0; i < worldSize; ++i) {
      std::srand(i + 1);
      accumulator += Eigen::Vector<T, Eigen::Dynamic>(m).setRandom();
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
  CHECK_CUDART(cudaFree(xDev));
}
}  // namespace

TEST_F(AllReduceNcclTestFixture, TestForwardF32) {
  TestNcclAllReduceT<float>(context, contextMpi, contextNccl);
}
TEST_F(AllReduceNcclTestFixture, TestForwardF64) {
  TestNcclAllReduceT<double>(context, contextMpi, contextNccl);
}

class AllReduceThreadStreamNcclTestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};
  int rank;
  dllm::ContextMpi contextMpi;
  dllm::ThreadStreamNccl *threadStreamNccl;

  AllReduceThreadStreamNcclTestFixture() {
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

  ~AllReduceThreadStreamNcclTestFixture() {
    delete threadStreamNccl;
    CHECK_MPI(MPI_Comm_free(&contextMpi.mpiComm));
  }
};

namespace {
template <typename T>
void TestThreadStreamNcclAllReduceT(dllm::ThreadStreamNccl &threadStreamNccl,
                                    dllm::ContextMpi &contextMpi) {
  const dllm::TensorIndexType m = 128;
  auto shapeX = cute::make_shape(m);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

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
      dllm::communication::AllReduce<dllm::communication::NCCL>::runInplace(
          tensorX, dllm::communication::SUM);
  threadStreamNccl.submit(std::move(task));
  tensorX->future->wait();
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
      accumulator += Eigen::Vector<T, Eigen::Dynamic>(m).setRandom();
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
  CHECK_CUDART(cudaFree(xDev));
}
}  // namespace

TEST_F(AllReduceThreadStreamNcclTestFixture, TestForwardF32) {
  TestThreadStreamNcclAllReduceT<float>(*threadStreamNccl, contextMpi);
}
TEST_F(AllReduceThreadStreamNcclTestFixture, TestForwardF64) {
  TestThreadStreamNcclAllReduceT<double>(*threadStreamNccl, contextMpi);
}