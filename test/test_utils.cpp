#include <ATen/Context.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/sum.h>
#include <gtest/gtest.h>

#include "compute/utils.h"
#include "memory/to_torch.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_cudart.h"

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

class TestUtilsFixture : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute tp{0, 2};
  dllm::ThreadStreamCudart stream{0};

  template <typename T>
  void TestCat(int size);

  template <typename T>
  void TestSum(int size);
};

template <typename T>
void TestUtilsFixture::TestCat(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  dllm::Tensor x1;
  dllm::Tensor x2;
  dllm::Tensor x3;
  dllm::Tensor y;
  dllm::compute::Utils::rand(tp, x1, {size, size, size}, option);
  dllm::compute::Utils::rand(tp, x2, {size, size, size}, option);
  dllm::compute::Utils::rand(tp, x3, {size, size, size}, option);
  dllm::compute::Utils::cat(tp, y, {x1, x2, x3}, -1);
  at::Tensor x1_torch, x2_torch, x3_torch, y_torch;
  dllm::memory::toTorch(stream, x1_torch, x1);
  x1.wait();
  dllm::memory::toTorch(stream, x2_torch, x2);
  x2.wait();
  dllm::memory::toTorch(stream, x3_torch, x3);
  x3.wait();
  dllm::memory::toTorch(stream, y_torch, y);
  y.wait();
  ASSERT_TRUE(at::allclose(y, at::cat({x1_torch, x2_torch, x3_torch}, -1)));
}

TEST_F(TestUtilsFixture, TestCatF32) { TestCat<float>(128); }
TEST_F(TestUtilsFixture, TestCatF64) { TestCat<double>(128); }

template <typename T>
void TestUtilsFixture::TestSum(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  dllm::Tensor x;
  dllm::Tensor y;
  dllm::compute::Utils::rand(tp, x, {size, size, size}, option);
  dllm::compute::Utils::sum(tp, y, x, 0);
  at::Tensor x_torch, y_torch;
  dllm::memory::toTorch(stream, x_torch, x);
  x.wait();
  dllm::memory::toTorch(stream, y_torch, y);
  y.wait();
  ASSERT_TRUE(at::allclose(y, at::sum(x_torch, 0)));
}

TEST_F(TestUtilsFixture, TestSumF32) { TestSum<float>(128); }
TEST_F(TestUtilsFixture, TestSumF64) { TestSum<double>(128); }
