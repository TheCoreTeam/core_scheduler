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

namespace dllm::test {
ThreadStreamNccl *getNcclStream();
}  // namespace dllm::test

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

// class AllReduceMPITestFixture : public ::testing::Test {
//  protected:
//   dllm::ContextMpi contextMpi{
//       [] {
//         int rank;
//         CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
//         return rank;
//       }(),
//       [] {
//         int commSize;
//         CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
//         return commSize;
//       }(),
//       MPI_COMM_WORLD};
//
//   dllm::ThreadStreamMpi stream{contextMpi};
//   dllm::ThreadStreamCudart copy{0};
//   dllm::ThreadPoolCompute tp{0, 3};
//
//   template <typename T>
//   void TestAllReduceT(int m);
// };
//
// template <typename T>
// void AllReduceMPITestFixture::TestAllReduceT(const int m) {
//   const at::Device device = at::kCPU;
//   const at::ScalarType dtype = TypeToTorch<T>::type;
//   const auto option = at::TensorOptions().dtype(dtype).device(device);
//   at::manual_seed(stream.rank() + 1);
//   auto x = dllm::Tensor::create();
//   {
//     auto task = dllm::compute::Utils::rand(x, {m}, option);
//     tp.submit(std::move(task));
//   }
//   {
//     auto task =
//         dllm::communication::AllReduce<dllm::communication::MPI>::runInplace(
//             x, dllm::communication::SUM);
//     stream.submit(std::move(task));
//   }
//
//   at::Tensor x_torch;
//   {
//     auto task = dllm::memory::toTorch(x_torch, x);
//     copy.submit(std::move(task));
//     x->wait();
//   }
//
//   auto accumulator = torch::zeros_like(x_torch);
//   if (contextMpi.mpiRank == 0) {
//     for (int i = 0; i < stream.commSize(); ++i) {
//       at::manual_seed(i + 1);
//       accumulator += torch::rand({m}, option);
//     }
//     ASSERT_TRUE(at::allclose(x, x_torch));
//   }
// }
//
// TEST_F(AllReduceMPITestFixture, TestForwardF32) { TestAllReduceT<float>(128);
// } TEST_F(AllReduceMPITestFixture, TestForwardF64) {
// TestAllReduceT<double>(128); }

class AllReduceNcclTestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};
  dllm::ThreadStreamCudart *copy;
  dllm::ThreadPoolCompute *tp;

  AllReduceNcclTestFixture() {
    copy = new dllm::ThreadStreamCudart{0};
    tp = new dllm::ThreadPoolCompute{0, 3};
    CHECK_CUDART(cudaSetDevice(0));
  }

  ~AllReduceNcclTestFixture() {
    delete tp;
    delete copy;
  }

  template <typename T>
  void TestAllReduceT(int m);
};

template <typename T>
void AllReduceNcclTestFixture::TestAllReduceT(const int m) {
  const auto stream = dllm::test::getNcclStream();
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  at::manual_seed(stream->rank() + 1);
  dllm::Tensor x;
  dllm::compute::Utils::rand(*tp, x, {m}, option);
  dllm::communication::AllReduce<dllm::communication::NCCL>::runInplace(
      *stream, {x}, dllm::communication::SUM);

  at::Tensor x_torch;
  dllm::memory::toTorch(*copy, x_torch, x);
  x.wait();

  auto accumulator = torch::zeros_like(x_torch);
  for (int i = 0; i < stream->commSize(); ++i) {
    at::manual_seed(i + 1);
    accumulator += torch::rand({m}, option);
  }
  GTEST_ASSERT_TRUE(at::allclose(x, x_torch));
}

TEST_F(AllReduceNcclTestFixture, TestForwardF32) { TestAllReduceT<float>(128); }
TEST_F(AllReduceNcclTestFixture, TestForwardF64) {
  TestAllReduceT<double>(128);
}
