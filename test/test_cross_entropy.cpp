#include <ATen/Context.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "compute/cross_entropy.h"
#include "compute/utils.h"
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

class TestCrossEntropyFixture : public ::testing::Test {
 protected:
  dllm::DynamicScheduler scheduler{0};

  template <typename T>
  void Test(int size);
};

template <typename T>
void TestCrossEntropyFixture::Test(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  auto x =
      dllm::compute::Utils::rand(scheduler, {size, 2 * size, 3 * size}, option);
  x = dllm::compute::Utils::view(scheduler, x, {-1, x.size(-1)});
  auto target = dllm::compute::Utils::randint(
      scheduler, 0, 3 * size, {size, 2 * size}, option.dtype(at::kLong));
  target = dllm::compute::Utils::view(scheduler, target, {-1});
  auto state = dllm::compute::CrossEntropy::init(scheduler);
  auto loss = dllm::compute::CrossEntropy::forward(scheduler, state, x, target);
  auto dx = dllm::compute::CrossEntropy::backward(scheduler, state);
  auto loss_ref_torch = dllm::memory::toTorch(scheduler, loss);
  loss.wait();
  auto x_torch = dllm::memory::toTorch(scheduler, x);
  x.wait();
  auto dx_torch = dllm::memory::toTorch(scheduler, dx);
  dx.wait();
  auto target_torch = dllm::memory::toTorch(scheduler, target);
  target.wait();
  x_torch.set_requires_grad(true);
  const auto loss_torch = at::cross_entropy_loss(x_torch, target_torch);
  loss_torch.backward();
  ASSERT_TRUE(at::allclose(loss_torch, loss_ref_torch));
  ASSERT_TRUE(at::allclose(x_torch.grad(), dx));
}

TEST_F(TestCrossEntropyFixture, TestF32) { Test<float>(128); }
TEST_F(TestCrossEntropyFixture, TestF64) { Test<double>(128); }
