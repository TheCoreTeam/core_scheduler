#include <communication/all_reduce.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/all_to_all.h"
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

// class AllToAllMPITestFixture : public ::testing::Test {
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
//   dllm::ThreadStreamMpi stream{contextMpi};
//   dllm::ThreadStreamCudart copy{0};
//   dllm::ThreadPoolCompute tp{0, 3};
//
//   template <typename T>
//   void TestAllToAllT(int blockSize);
// };
//
// template <typename T>
// void AllToAllMPITestFixture::TestAllToAllT(const int blockSize) {
//   const at::Device device = at::kCPU;
//   const at::ScalarType dtype = TypeToTorch<T>::type;
//   const auto option = at::TensorOptions().dtype(dtype).device(device);
//   const int m = blockSize * stream.commSize();
//   at::manual_seed(stream.rank() + 1);
//   auto x = dllm::Tensor::create();
//   {
//     auto task = dllm::compute::Utils::rand(x, {m}, option);
//     tp.submit(std::move(task));
//   }
//   {
//     auto task =
//         dllm::communication::AllToAll<dllm::communication::MPI>::runInplace(x);
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
//   for (int i = 0; i < stream.commSize(); ++i) {
//     at::manual_seed(i + 1);
//     auto full_random = torch::rand({m}, option);
//     accumulator.narrow(0, i * blockSize, blockSize)
//         .copy_(full_random.narrow(0, stream.rank() * blockSize, blockSize));
//   }
//   ASSERT_TRUE(at::allclose(accumulator, x_torch));
// }
//
// TEST_F(AllToAllMPITestFixture, TestForwardF32) { TestAllToAllT<float>(128); }
// TEST_F(AllToAllMPITestFixture, TestForwardF64) { TestAllToAllT<double>(128);
// }

class AllToAllNCCLTestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};
  dllm::ThreadStreamCudart *copy;
  dllm::ThreadPoolCompute *tp;

  AllToAllNCCLTestFixture() {
    copy = new dllm::ThreadStreamCudart{0};
    tp = new dllm::ThreadPoolCompute{0, 3};
    CHECK_CUDART(cudaSetDevice(0));
  }

  ~AllToAllNCCLTestFixture() {
    delete tp;
    delete copy;
  }

  template <typename T>
  void TestlAllToAllT(int blockSize);
};

template <typename T>
void AllToAllNCCLTestFixture::TestlAllToAllT(const int blockSize) {
  const auto stream = dllm::test::getNcclStream();
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  at::manual_seed(stream->rank() + 1);
  std::vector<std::shared_ptr<const dllm::ReadOnlyTensor>> s;
  s.reserve(stream->commSize());
  for (int i = 0; i < stream->commSize(); ++i) {
    auto t = dllm::Tensor::create();
    dllm::compute::Utils::rand(*tp, t, {blockSize}, option);
    s.push_back(t);
  }
  std::vector<std::shared_ptr<dllm::Tensor>> r;
  r.reserve(stream->commSize());
  for (int i = 0; i < stream->commSize(); ++i) {
    auto t = dllm::Tensor::create();
    dllm::compute::Utils::empty(*tp, t, {blockSize}, option);
    r.push_back(t);
  }
  dllm::communication::AllToAll<dllm::communication::NCCL>::run(*stream, r, s);

  std::vector<at::Tensor> r_torch;
  r_torch.resize(r.size());
  for (int i = 0; i < stream->commSize(); ++i) {
    dllm::memory::toTorch(*copy, r_torch[i], r[i]);
    r[i]->wait();
  }

  for (int i = 0; i < stream->commSize(); ++i) {
    at::manual_seed(i + 1);
    at::Tensor full_random;
    for (int j = 0; j <= stream->rank(); ++j) {
      full_random = torch::rand({blockSize}, option);
    }
    ASSERT_TRUE(at::allclose(r_torch[i], full_random));
  }
}

TEST_F(AllToAllNCCLTestFixture, TestForwardF32) { TestlAllToAllT<float>(128); }
TEST_F(AllToAllNCCLTestFixture, TestForwardF64) { TestlAllToAllT<double>(128); }
