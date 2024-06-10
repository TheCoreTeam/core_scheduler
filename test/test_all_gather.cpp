#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/all_gather.h"
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

class AllGatherNCCLTestFixture : public ::testing::Test {
 protected:
  dllm::DynamicScheduler scheduler{0};

  AllGatherNCCLTestFixture() { CHECK_CUDART(cudaSetDevice(0)); }

  template <typename T>
  void TestlAllToAllT(int blockSize);
};

template <typename T>
void AllGatherNCCLTestFixture::TestlAllToAllT(const int blockSize) {
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  const auto comm =
      dllm::communication::getCommWorld(dllm::communication::NCCL);
  at::manual_seed(comm.getRank() + 1);
  dllm::Tensor x;
  dllm::compute::Utils::rand(scheduler, x, {blockSize}, option);
  std::vector<dllm::Tensor> r;
  r.reserve(comm.getSize());
  for (int i = 0; i < comm.getSize(); ++i) {
    dllm::Tensor t;
    dllm::compute::Utils::empty(scheduler, t, {blockSize}, option);
    r.push_back(t);
  }
  dllm::communication::AllGather::run(scheduler, comm, {r}, {x});
  std::vector<at::Tensor> r_torch;
  r_torch.resize(r.size());
  for (int i = 0; i < r.size(); ++i) {
    dllm::memory::toTorch(scheduler, r_torch[i], r[i]);
    r[i].wait();
  }

  for (int i = 0; i < comm.getSize(); ++i) {
    at::manual_seed(i + 1);
    auto full_random = torch::rand({blockSize}, option);
    ASSERT_TRUE(at::allclose(r_torch[i], full_random));
  }
}

TEST_F(AllGatherNCCLTestFixture, TestForwardF32) { TestlAllToAllT<float>(128); }
TEST_F(AllGatherNCCLTestFixture, TestForwardF64) {
  TestlAllToAllT<double>(128);
}
