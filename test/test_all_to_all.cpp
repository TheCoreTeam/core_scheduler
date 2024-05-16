#include <gtest/gtest.h>
#include <mpi.h>

#include <Eigen/Dense>

#include "communication/mpi/all_to_all.h"
#include "communication/nccl/all_to_all.h"
#include "dtensor_mpi.h"
#include "dtensor_nccl.h"
#include "logger.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_mpi.h"
#include "threading/thread_stream_nccl.h"

class AllToAllMPITestFixture : public ::testing::Test {
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
void TestAllToAllT(const dllm::ContextMpi &contextMpi) {
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
      dllm::communication::AllToAll<dllm::communication::MPI>::runInplace(
          tensorX);
  tensorX.reset();
  task(&contextMpi);

  Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
  {
    int worldSize;
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
    accumulator.setZero();
    for (int i = 0; i < worldSize; ++i) {
      std::srand(i + 1);
      accumulator.segment(i * blockSize, blockSize) =
          Eigen::Vector<T, Eigen::Dynamic>(m).setRandom().segment(
              contextMpi.mpiRank * blockSize, blockSize);
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
}
}  // namespace

TEST_F(AllToAllMPITestFixture, TestForwardF32) {
  TestAllToAllT<float>(contextMpi);
}
TEST_F(AllToAllMPITestFixture, TestForwardF64) {
  TestAllToAllT<double>(contextMpi);
}

class AllToAllMPIThreadPoolComputeTestFixture : public ::testing::Test {
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
void TestThreadPoolComputeAllToAllT(dllm::ThreadStreamMpi &threadStreamMpi,
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
      dllm::communication::AllToAll<dllm::communication::MPI>::runInplace(
          tensorX);
  threadStreamMpi.submit(std::move(task));
  tensorX->future->wait();

  Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
  {
    int worldSize;
    CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
    accumulator.setZero();
    for (int i = 0; i < worldSize; ++i) {
      std::srand(i + 1);
      accumulator.segment(i * blockSize, blockSize) =
          Eigen::Vector<T, Eigen::Dynamic>(m).setRandom().segment(
              contextMpi.mpiRank * blockSize, blockSize);
    }
    ASSERT_NEAR(
        accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
        1e-4);
  }
}
}  // namespace

TEST_F(AllToAllMPIThreadPoolComputeTestFixture, TestForwardF32) {
  TestThreadPoolComputeAllToAllT<float>(threadStreamMpi, contextMpi);
}
TEST_F(AllToAllMPIThreadPoolComputeTestFixture, TestForwardF64) {
  TestThreadPoolComputeAllToAllT<double>(threadStreamMpi, contextMpi);
}

// class AllToAllNcclTestFixture : public ::testing::Test {
//  protected:
//   int rank;
//   dllm::ContextMpi contextMpi;
//   dllm::ContextNccl contextNccl;
//
//   void SetUp() override {
//     int processesPerNode;
//     CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
//                                   MPI_INFO_NULL, &contextMpi.mpiComm));
//     CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &processesPerNode));
//     CHECK_MPI(MPI_Comm_rank(contextMpi.mpiComm, &contextMpi.mpiRank));
//     ncclUniqueId id;
//     if (contextMpi.mpiRank == 0) {
//       CHECK_NCCL(ncclGetUniqueId(&id));
//     }
//     CHECK_MPI(
//         MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0,
//         contextMpi.mpiComm));
//     contextNccl.ncclRank = contextMpi.mpiRank;
//     CHECK_CUDART(cudaSetDevice(contextMpi.mpiRank));
//     CHECK_NCCL(ncclCommInitRank(&contextNccl.ncclComm, processesPerNode, id,
//                                 contextMpi.mpiRank));
//   }
//
//   void TearDown() override {
//     CHECK_NCCL(ncclCommDestroy(contextNccl.ncclComm));
//     CHECK_MPI(MPI_Comm_free(&contextMpi.mpiComm));
//   }
// };
//
// namespace {
// template <typename T>
// void TestNcclAllToAllT(const dllm::ContextMpi &contextMpi,
//                        const dllm::ContextNccl &contextNccl) {
//   int commSize;
//   CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &commSize));
//   const int blockSize = 128;
//   const dllm::TensorIndexType m = blockSize * commSize;
//   const auto shapeX = cute::make_shape(m);
//   const auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
//
//   Eigen::Vector<T, Eigen::Dynamic> x(m);
//   std::srand(contextMpi.mpiRank + 1);
//   x.setRandom();
//
//   T *xDev;
//   CHECK_CUDART(cudaMalloc(&xDev, sizeof(T) * cute::size(layoutX)));
//   CHECK_CUDART(cudaMemcpy(xDev, x.data(), sizeof(T) * cute::size(layoutX),
//                           cudaMemcpyHostToDevice));
//   CHECK_CUDART(cudaDeviceSynchronize());
//
//   auto tensorX = std::make_shared<dllm::Tensor1D>(
//       dllm::Tensor1D{xDev, layoutX, dllm::toDtype<T>(), dllm::CUDA});
//
//   auto task =
//       dllm::communication::AllToAll<dllm::communication::NCCL>::runInplace(
//           tensorX);
//   task(&contextNccl);
//   tensorX->future->wait();
//
//   CHECK_CUDART(cudaMemcpy(x.data(), xDev, sizeof(T) * cute::size(layoutX),
//                           cudaMemcpyDeviceToHost));
//
//   Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
//   if (contextMpi.mpiRank == 0) {
//     int worldSize;
//     CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
//     accumulator.setZero();
//     for (int i = 0; i < worldSize; ++i) {
//       std::srand(i + 1);
//       accumulator.segment(i * blockSize, blockSize) =
//           Eigen::Vector<T, Eigen::Dynamic>(m).setRandom().segment(i *
//           blockSize,
//                                                                   blockSize);
//     }
//     ASSERT_NEAR(
//         accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
//         1e-4);
//   }
//   CHECK_CUDART(cudaFree(xDev));
// }
// }  // namespace
//
// TEST_F(AllToAllNcclTestFixture, TestForwardF32) {
//   TestNcclAllToAllT<float>(contextMpi, contextNccl);
// }
// TEST_F(AllToAllNcclTestFixture, TestForwardF64) {
//   TestNcclAllToAllT<double>(contextMpi, contextNccl);
// }
//
// class AllToAllThreadStreamNcclTestFixture : public ::testing::Test {
//  protected:
//   dllm::ContextCompute context{};
//   int rank;
//   dllm::ContextMpi contextMpi;
//   dllm::ThreadStreamNccl *threadStreamNccl;
//
//   AllToAllThreadStreamNcclTestFixture() {
//     int processesPerNode;
//     CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
//                                   MPI_INFO_NULL, &contextMpi.mpiComm));
//     CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &processesPerNode));
//     CHECK_MPI(MPI_Comm_rank(contextMpi.mpiComm, &contextMpi.mpiRank));
//     ncclUniqueId id;
//     if (contextMpi.mpiRank == 0) {
//       CHECK_NCCL(ncclGetUniqueId(&id));
//     }
//     CHECK_MPI(
//         MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0,
//         contextMpi.mpiComm));
//     threadStreamNccl = new dllm::ThreadStreamNccl{
//         id, processesPerNode, contextMpi.mpiRank, contextMpi.mpiRank};
//     CHECK_CUDART(cudaSetDevice(contextMpi.mpiRank));
//   }
//
//   ~AllToAllThreadStreamNcclTestFixture() {
//     delete threadStreamNccl;
//     CHECK_MPI(MPI_Comm_free(&contextMpi.mpiComm));
//   }
// };
//
// namespace {
// template <typename T>
// void TestThreadStreamNcclAllToAllT(dllm::ThreadStreamNccl &threadStreamNccl,
//                                    dllm::ContextMpi &contextMpi) {
//   int commSize;
//   CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &commSize));
//   const int blockSize = 128;
//   const dllm::TensorIndexType m = blockSize * commSize;
//   const auto shapeX = cute::make_shape(m);
//   const auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
//
//   Eigen::Vector<T, Eigen::Dynamic> x(m);
//   std::srand(contextMpi.mpiRank + 1);
//   x.setRandom();
//
//   T *xDev;
//   CHECK_CUDART(cudaMalloc(&xDev, sizeof(T) * cute::size(layoutX)));
//   CHECK_CUDART(cudaMemcpy(xDev, x.data(), sizeof(T) * cute::size(layoutX),
//                           cudaMemcpyHostToDevice));
//   CHECK_CUDART(cudaDeviceSynchronize());
//
//   auto tensorX = std::make_shared<dllm::Tensor1D>(
//       dllm::Tensor1D{xDev, layoutX, dllm::toDtype<T>(), dllm::CUDA});
//
//   auto task =
//       dllm::communication::AllToAll<dllm::communication::NCCL>::runInplace(
//           tensorX);
//   threadStreamNccl.submit(std::move(task));
//   tensorX->future->wait();
//   CHECK_CUDART(cudaMemcpy(x.data(), xDev, sizeof(T) * cute::size(layoutX),
//                           cudaMemcpyDeviceToHost));
//   CHECK_CUDART(cudaDeviceSynchronize());
//
//   Eigen::Vector<T, Eigen::Dynamic> accumulator(m);
//   if (contextMpi.mpiRank == 0) {
//     int worldSize;
//     CHECK_MPI(MPI_Comm_size(contextMpi.mpiComm, &worldSize));
//     accumulator.setZero();
//     for (int i = 0; i < worldSize; ++i) {
//       std::srand(i + 1);
//       accumulator.segment(i * blockSize, blockSize) =
//           Eigen::Vector<T, Eigen::Dynamic>(m).setRandom().segment(i *
//           blockSize,
//                                                                   blockSize);
//     }
//     ASSERT_NEAR(
//         accumulator.array().abs().maxCoeff() - x.array().abs().maxCoeff(), 0,
//         1e-4);
//   }
//   CHECK_CUDART(cudaFree(xDev));
// }
// }  // namespace
//
// TEST_F(AllToAllThreadStreamNcclTestFixture, TestForwardF32) {
//   TestThreadStreamNcclAllToAllT<float>(*threadStreamNccl, contextMpi);
// }
// TEST_F(AllToAllThreadStreamNcclTestFixture, TestForwardF64) {
//   TestThreadStreamNcclAllToAllT<double>(*threadStreamNccl, contextMpi);
// }