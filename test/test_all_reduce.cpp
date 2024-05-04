#include <dtensor_mpi.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <Eigen/Dense>

#include "communication/mpi/all_reduce.h"
#include "communication/nccl/all_reduce.h"
#include "logger.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_mpi.h"

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
                    const dllm::ContextMpi contextMpi) {
  const int m = 128;
  auto shapeX = cute::make_shape(m);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

  Eigen::Vector<T, Eigen::Dynamic> x(m);

  auto tensorX = std::make_shared<dllm::DTensor1D<dllm::communication::MPI>>(
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
  dllm::ThreadPoolCompute threadPool{0, 3};
  dllm::ThreadStreamMpi threadStreamMpi{contextMpi};
};

namespace {
template <typename T>
void TestThreadPoolComputeAllReduceT(dllm::ThreadPoolCompute &threadPool,
                                     dllm::ThreadStreamMpi &threadStreamMpi,
                                     dllm::ContextMpi contextMpi) {
  const int m = 128;
  auto shapeX = cute::make_shape(m);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});

  Eigen::Vector<T, Eigen::Dynamic> x(m);

  auto tensorX = std::make_shared<dllm::DTensor1D<dllm::communication::MPI>>(
      dllm::Tensor1D{x.data(), layoutX, dllm::toDtype<T>(), dllm::CUDA});

  std::srand(contextMpi.mpiRank + 1);
  x.setRandom();

  auto task =
      dllm::communication::AllReduce<dllm::communication::MPI>::runInplace(
          tensorX, dllm::communication::SUM);
  tensorX.reset();
  auto future = threadStreamMpi.submit(std::move(task));
  future->wait();

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
  TestThreadPoolComputeAllReduceT<float>(threadPool, threadStreamMpi,
                                         contextMpi);
}
TEST_F(AllReduceMPIThreadPoolComputeTestFixture, TestForwardF64) {
  TestThreadPoolComputeAllReduceT<double>(threadPool, threadStreamMpi,
                                          contextMpi);
}
