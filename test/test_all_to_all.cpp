#include <communication/all_reduce.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/all_to_all.h"
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

class AllToAllNCCLTestFixture : public ::testing::Test {
 protected:
  dllm::communication::Comm comm{
      dllm::communication::getCommWorld(dllm::communication::NCCL)};
  dllm::DynamicScheduler scheduler{static_cast<int>(comm.getRank())};

  AllToAllNCCLTestFixture() { CHECK_CUDART(cudaSetDevice(comm.getRank())); }

  template <typename T>
  void TestlAllToAllT(int blockSize);
};

template <typename T>
void AllToAllNCCLTestFixture::TestlAllToAllT(const int blockSize) {
  const at::Device device(at::kCUDA, comm.getRank());
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  at::manual_seed(comm.getRank() + 1);
  std::vector<dllm::ReadOnlyTensor> s;
  s.reserve(comm.getSize());
  for (int i = 0; i < comm.getSize(); ++i) {
    dllm::Tensor t;
    dllm::compute::Utils::rand(scheduler, t, {blockSize}, option);
    s.push_back(t);
  }
  std::vector<dllm::Tensor> r;
  r.reserve(comm.getSize());
  for (int i = 0; i < comm.getSize(); ++i) {
    dllm::Tensor t;
    dllm::compute::Utils::empty(scheduler, t, {blockSize}, option);
    r.push_back(t);
  }
  dllm::communication::AllToAll::run(scheduler, comm, r, s);

  std::vector<at::Tensor> r_torch;
  r_torch.resize(r.size());
  for (int i = 0; i < comm.getSize(); ++i) {
    dllm::memory::toTorch(scheduler, r_torch[i], r[i]);
    r[i].wait();
  }

  for (int i = 0; i < comm.getSize(); ++i) {
    at::manual_seed(i + 1);
    at::Tensor full_random;
    for (int j = 0; j <= comm.getRank(); ++j) {
      full_random = torch::rand({blockSize}, option);
    }
    ASSERT_TRUE(at::allclose(r_torch[i], full_random));
  }
}

TEST_F(AllToAllNCCLTestFixture, TestForwardF32) { TestlAllToAllT<float>(128); }
TEST_F(AllToAllNCCLTestFixture, TestForwardF64) { TestlAllToAllT<double>(128); }
