#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/all_reduce.h"
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

class AllReduceNcclTestFixture : public ::testing::Test {
 protected:
  dllm::communication::Comm comm{
      dllm::communication::getCommWorld(dllm::communication::NCCL)};
  dllm::DynamicScheduler scheduler{static_cast<int>(comm.getRank())};

  AllReduceNcclTestFixture() { CHECK_CUDART(cudaSetDevice(comm.getRank())); }

  template <typename T>
  void TestAllReduceT(int m);
};

template <typename T>
void AllReduceNcclTestFixture::TestAllReduceT(const int m) {
  const at::Device device(at::kCUDA, comm.getRank());
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  at::manual_seed(comm.getRank() + 1);
  dllm::Tensor x, y;
  dllm::compute::Utils::rand(scheduler, x, {m}, option);
  dllm::compute::Utils::rand(scheduler, y, {m}, option);
  dllm::communication::AllReduce::runInplace(scheduler, comm, {x, y},
                                             dllm::communication::SUM);

  at::Tensor x_torch, y_torch;
  dllm::memory::toTorch(scheduler, x_torch, x);
  dllm::memory::toTorch(scheduler, y_torch, y);
  x.wait();
  y.wait();

  auto accumulator_x = torch::zeros_like(x_torch);
  auto accumulator_y = torch::zeros_like(y_torch);
  for (int i = 0; i < comm.getSize(); ++i) {
    at::manual_seed(i + 1);
    accumulator_x += torch::rand({m}, option);
    accumulator_y += torch::rand({m}, option);
  }
  GTEST_ASSERT_TRUE(at::allclose(x, x_torch));
  GTEST_ASSERT_TRUE(at::allclose(y, y_torch));
}

TEST_F(AllReduceNcclTestFixture, TestForwardF32) { TestAllReduceT<float>(128); }
TEST_F(AllReduceNcclTestFixture, TestForwardF64) {
  TestAllReduceT<double>(128);
}
