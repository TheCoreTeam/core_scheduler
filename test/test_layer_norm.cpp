#include <ATen/Context.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/cat.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/nn/modules/normalization.h>

#include "compute/layer_norm.h"
#include "compute/utils.h"
#include "memory/to_torch.h"
#include "module/layer_norm.h"
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

class TestLayerNormFixture : public ::testing::Test {
 protected:
  dllm::DynamicScheduler scheduler{0};

  template <typename T>
  void TestFunctional(int size);

  template <typename T>
  void TestModule(int size);
};

template <typename T>
void TestLayerNormFixture::TestFunctional(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);

  auto state = dllm::compute::LayerNorm::init(
      scheduler,
      dllm::compute::LayerNorm::Options{{3 * size}}.device(device).dtype(
          dtype));
  auto x =
      dllm::compute::Utils::rand(scheduler, {size, 2 * size, 3 * size}, option);
  auto y = dllm::compute::LayerNorm::forward(scheduler, state, x);
  auto dy = dllm::compute::Utils::rand_like(scheduler, y);
  auto dx = dllm::compute::LayerNorm::backward(scheduler, state, dy);

  auto x_torch = dllm::memory::toTorch(scheduler, x);
  x.wait();
  auto y_ref_torch = dllm::memory::toTorch(scheduler, y);
  y.wait();
  auto dy_torch = dllm::memory::toTorch(scheduler, dy);
  dy.wait();
  x_torch.set_requires_grad(true);
  torch::nn::LayerNorm ln{torch::nn::LayerNormOptions({3 * size})};
  ln->to(device, dtype);
  auto y_torch = ln->forward(x_torch);
  y_torch.backward(dy_torch);
  ASSERT_TRUE(at::allclose(y_torch, y));
  ASSERT_TRUE(at::allclose(x_torch.grad(), dx));
  ASSERT_TRUE(at::allclose(ln->weight.grad(), state->forward.grad_weight));
  ASSERT_TRUE(at::allclose(ln->bias.grad(), state->forward.grad_bias));
}

TEST_F(TestLayerNormFixture, TestFunctionalF32) { TestFunctional<float>(128); }
TEST_F(TestLayerNormFixture, TestFunctionalF64) { TestFunctional<double>(128); }

template <typename T>
void TestLayerNormFixture::TestModule(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);

  dllm::module::LayerNorm lnOurs{
      scheduler,
      dllm::module::LayerNorm::Options{{3 * size}}.device(device).dtype(dtype)};
  auto x =
      dllm::compute::Utils::rand(scheduler, {size, 2 * size, 3 * size}, option);
  auto y = lnOurs->forward(scheduler, x);
  auto dy = dllm::compute::Utils::rand_like(scheduler, y);
  auto dx = lnOurs->backward(scheduler, dy);

  auto x_torch = dllm::memory::toTorch(scheduler, x);
  x.wait();
  auto y_ref_torch = dllm::memory::toTorch(scheduler, y);
  y.wait();
  auto dy_torch = dllm::memory::toTorch(scheduler, dy);
  dy.wait();
  x_torch.set_requires_grad(true);
  torch::nn::LayerNorm ln{torch::nn::LayerNormOptions({3 * size})};
  ln->to(device, dtype);
  auto y_torch = ln->forward(x_torch);
  y_torch.backward(dy_torch);
  ASSERT_TRUE(at::allclose(y_torch, y));
  ASSERT_TRUE(at::allclose(x_torch.grad(), dx));
  ASSERT_TRUE(
      at::allclose(ln->weight.grad(), lnOurs->state()->forward.grad_weight));
  ASSERT_TRUE(
      at::allclose(ln->bias.grad(), lnOurs->state()->forward.grad_bias));
}

TEST_F(TestLayerNormFixture, TestTestModuleF32) { TestModule<float>(128); }
TEST_F(TestLayerNormFixture, TestTestModuleF64) { TestModule<double>(128); }
