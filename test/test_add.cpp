#include <cuda_runtime.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <fstream>

#include "compute/add.h"
#include "threading/thread_pool_compute.h"

template <typename T>
struct TypeTrait;

template <>
struct TypeTrait<float> {
  static constexpr at::ScalarType dtype = torch::kFloat;
  static constexpr auto label = "float";
};

template <>
struct TypeTrait<nv_half> {
  static constexpr at::ScalarType dtype = torch::kHalf;
  static constexpr auto label = "half";
};

template <>
struct TypeTrait<double> {
  static constexpr at::ScalarType dtype = torch::kDouble;
  static constexpr auto label = "double";
};

template <typename Element>
class TestDLLMResidual : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute tp{0, 2};

  const int M = 32;
  const int N = 64;
  const int K = 128;

  torch::Device device = torch::kCUDA;
  torch::Dtype dtype = TypeTrait<Element>::dtype;

  at::Tensor A;
  at::Tensor B;
  at::Tensor output;

  at::Tensor grad_output;
  at::Tensor grad_A;
  at::Tensor grad_B;

  void SetUp() override;

  void TestForwardRoutine();
};

template <typename Element>
void TestDLLMResidual<Element>::SetUp() {
  torch::manual_seed(2024);

  A = torch::rand(
      {M, N, K},
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));

  B = torch::rand(
      {M, N, K},
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));

  output = A + B;

  grad_output = torch::rand_like(A);
}

template <typename Element>
void TestDLLMResidual<Element>::TestForwardRoutine() {
  auto output_compute =
      std::make_shared<dllm::Tensor>(torch::empty_like(output));
  auto task = dllm::compute::Add::forward(output_compute,
                                          std::make_shared<dllm::Tensor>(A),
                                          std::make_shared<dllm::Tensor>(B));
  tp.submit(std::move(task));
  output_compute->wait();
  ASSERT_TRUE(torch::allclose(output, output_compute->tensor()));
}

using ElementTypes = ::testing::Types<float, nv_half, double>;
TYPED_TEST_SUITE(TestDLLMResidual, ElementTypes);

TYPED_TEST(TestDLLMResidual, TestVaryingElementTypes) {
  this->TestForwardRoutine();
}
