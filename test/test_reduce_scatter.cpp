#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/reduce_scatter.h"
#include "compute/utils.h"
#include "logger.h"
#include "memory/to_torch.h"
#include "threading/dynamic_scheduler.h"

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

class ReduceScatterNCCLTestFixture : public ::testing::Test {
 protected:
  dllm::communication::Comm comm{dllm::communication::getCommWorld(dllm::communication::NCCL)};
  dllm::DynamicScheduler scheduler{static_cast<int>(comm.getRank())};

  ReduceScatterNCCLTestFixture() { CHECK_CUDART(cudaSetDevice(comm.getRank())); }

  template <typename T>
  void TestlAllToAllT(int blockSize);
};

template <typename T>
void ReduceScatterNCCLTestFixture::TestlAllToAllT(const int blockSize) {
  const at::Device device(at::kCUDA, comm.getRank());
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  at::manual_seed(comm.getRank() + 1);
  const int m = blockSize * comm.getSize();
  std::vector<dllm::ReadOnlyTensor> vs;
  vs.reserve(comm.getSize());
  for (int i = 0; i < comm.getSize(); ++i) {
    dllm::Tensor t;
    dllm::compute::Utils::rand(scheduler, t, {blockSize}, option);
    vs.push_back(t);
  }
  dllm::Tensor r;
  dllm::compute::Utils::empty(scheduler, r, {blockSize}, option);
  dllm::communication::ReduceScatter::run(scheduler, comm, {r}, {vs},
                                          dllm::communication::SUM);

  at::Tensor x_torch;
  dllm::memory::toTorch(scheduler, x_torch, r);
  r.wait();

  auto accumulator = torch::zeros({m}, option);
  for (int i = 0; i < comm.getSize(); ++i) {
    at::manual_seed(i + 1);
    for (int j = 0; j < comm.getSize(); ++j) {
      const auto full_random = torch::rand({blockSize}, option);
      accumulator.narrow(0, j * blockSize, blockSize) += full_random;
    }
  }
  ASSERT_TRUE(at::allclose(
      accumulator.narrow(0, comm.getRank() * blockSize, blockSize), x_torch));
}

TEST_F(ReduceScatterNCCLTestFixture, TestForwardF32) {
  TestlAllToAllT<float>(128);
}
TEST_F(ReduceScatterNCCLTestFixture, TestForwardF64) {
  TestlAllToAllT<double>(128);
}
