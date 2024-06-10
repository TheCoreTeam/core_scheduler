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
  dllm::Tensor loss;
  dllm::Tensor x;
  dllm::Tensor dx;
  dllm::Tensor target;
  dllm::compute::Utils::rand(scheduler, x, {size, 2 * size, 3 * size}, option);
  dllm::compute::Utils::view(scheduler, x, x, {-1, x.size(-1)});
  dllm::compute::Utils::randint(scheduler, target, 0, 3 * size,
                                {size, 2 * size}, option.dtype(at::kLong));
  dllm::compute::Utils::view(scheduler, target, target, {-1});
  std::shared_ptr<dllm::compute::CrossEntropy::State> state;
  dllm::compute::CrossEntropy::init(scheduler, state);
  dllm::compute::CrossEntropy::forward(scheduler, state, loss, x, target);
  dllm::compute::CrossEntropy::backward(scheduler, state, dx);
  at::Tensor loss_ref_torch, x_torch, dx_torch, target_torch;
  dllm::memory::toTorch(scheduler, loss_ref_torch, loss);
  loss.wait();
  dllm::memory::toTorch(scheduler, x_torch, x);
  x.wait();
  dllm::memory::toTorch(scheduler, dx_torch, dx);
  dx.wait();
  dllm::memory::toTorch(scheduler, target_torch, target);
  target.wait();
  x_torch.set_requires_grad(true);
  const auto loss_torch = at::cross_entropy_loss(x_torch, target_torch);
  loss_torch.backward();
  ASSERT_TRUE(at::allclose(loss_torch, loss_ref_torch));
  ASSERT_TRUE(at::allclose(x_torch.grad(), dx));
}

TEST_F(TestCrossEntropyFixture, TestF32) { Test<float>(128); }
TEST_F(TestCrossEntropyFixture, TestF64) { Test<double>(128); }
