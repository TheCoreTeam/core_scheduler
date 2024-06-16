#include <ATen/Context.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include "compute/utils.h"
#include "memory/to_torch.h"
#include "module/linear.h"
#include "optimizer/adamw.h"
#include "tensor.h"
#include "threading/dynamic_scheduler.h"

template <typename T>
struct TypeToTorch;

template <>
struct TypeToTorch<float> {
  using Type = float;
  static const at::ScalarType type = torch::kFloat;
};

template <>
struct TypeToTorch<nv_half> {
  using Type = c10::Half;
  static const at::ScalarType type = torch::kHalf;
};

template <>
struct TypeToTorch<double> {
  using Type = double;
  static const at::ScalarType type = torch::kDouble;
};

class TestDLLMAdamW : public ::testing::Test {
 protected:
  dllm::DynamicScheduler scheduler{0};

  template <typename Element>
  void TestFunctionalRoutine(int size);

  template <typename Element>
  void TestModuleRoutine(int size);
};

template <typename Element>
void TestDLLMAdamW::TestFunctionalRoutine(const int size) {
  const double lr = 1e-3;
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double eps = 1e-8;
  const double weight_decay = 1e-2;
  const long t = 0;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  torch::manual_seed(1);
  auto x = dllm::compute::Utils::rand(scheduler, {size}, option);
  auto state = dllm::optimizer::AdamW::init(scheduler, x,
                                            dllm::optimizer::AdamW::Options{}
                                                .lr(lr)
                                                .beta1(beta1)
                                                .beta2(beta2)
                                                .eps(eps)
                                                .weight_decay(weight_decay)
                                                .t(t));
  auto dx = dllm::compute::Utils::rand(scheduler, {size}, option);
  dllm::optimizer::AdamW::step(scheduler, state, x, dx);
  auto x_torch = dllm::memory::toTorch(scheduler, x);
  x.wait();
  auto dx_torch = dllm::memory::toTorch(scheduler, dx);
  dx.wait();
  torch::manual_seed(1);
  x_torch = torch::rand_like(x_torch);
  auto m_torch = torch::zeros_like(x_torch);
  auto v_torch = torch::zeros_like(x_torch);

  x_torch = x_torch - lr * weight_decay * x_torch;
  m_torch = beta1 * m_torch + (1 - beta1) * dx_torch;
  v_torch = beta2 * v_torch + (1 - beta2) * dx_torch.square();
  auto m_hat = m_torch / (1 - std::pow(beta1, t + 1));
  auto v_hat = v_torch / (1 - std::pow(beta2, t + 1));
  x_torch = x_torch - lr * m_hat / (v_hat.sqrt() + eps);

  ASSERT_TRUE(torch::allclose(x_torch, x));
  ASSERT_TRUE(torch::allclose(m_torch, state->tensors.m));
  ASSERT_TRUE(torch::allclose(v_torch, state->tensors.v));
}

TEST_F(TestDLLMAdamW, TestFunctionalF32_128) {
  TestFunctionalRoutine<float>(128);
}

TEST_F(TestDLLMAdamW, TestFunctionalF32_512) {
  TestFunctionalRoutine<float>(512);
}

TEST_F(TestDLLMAdamW, TestFunctionalF32_1024) {
  TestFunctionalRoutine<float>(1024);
}

template <typename Element>
void TestDLLMAdamW::TestModuleRoutine(const int size) {
  const double lr = 1e-3;
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double eps = 1e-8;
  const double weight_decay = 1e-2;
  const long t = 0;
  const int m = size, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  auto x = dllm::compute::Utils::randn(scheduler, {m, s, k}, option);
  dllm::module::Linear fc{
      scheduler,
      dllm::module::Linear::Options{k, n}.bias(false).device(device).dtype(
          dtype)};
  auto y = fc->forward(scheduler, x);
  auto yGrad = dllm::compute::Utils::randn_like(scheduler, y);
  auto dx = fc->backward(scheduler, yGrad);
  dx.wait();
  fc->state()->forward.grad_weight.wait();
  auto xRef = dllm::memory::toTorch(scheduler, x);
  x.wait();
  xRef.requires_grad_(true);
  auto wRef = dllm::memory::toTorch(scheduler, fc->state()->forward.weight);
  fc->state()->forward.weight.wait();
  wRef.requires_grad_(true);
  auto fcTorch = torch::nn::Linear{torch::nn::LinearOptions{k, n}.bias(false)};
  fcTorch->to(device, dtype);
  fcTorch->weight.data().copy_(wRef);
  torch::optim::AdamW optimTorch{fcTorch->parameters(),
                                 torch::optim::AdamWOptions{}
                                     .lr(lr)
                                     .betas({beta1, beta2})
                                     .eps(eps)
                                     .weight_decay(weight_decay)};
  auto yRef = fcTorch->forward(xRef);
  auto yGradRef = dllm::memory::toTorch(scheduler, yGrad);
  yGrad.wait();
  optimTorch.zero_grad();
  yRef.backward(yGradRef);

  ASSERT_TRUE(torch::allclose(y, yRef));
  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
  ASSERT_TRUE(torch::allclose(fc->state()->forward.grad_weight,
                              fcTorch->weight.grad()));

  optimTorch.step();

  dllm::optimizer::AdamW::init(scheduler, fc,
                               dllm::optimizer::AdamW::Options{}
                                   .lr(lr)
                                   .beta1(beta1)
                                   .beta2(beta2)
                                   .eps(eps)
                                   .weight_decay(weight_decay)
                                   .t(t));
  dllm::optimizer::AdamW::step(scheduler, fc);

  ASSERT_TRUE(torch::allclose(fc->state()->forward.weight, fcTorch->weight));
}

TEST_F(TestDLLMAdamW, TestModuleF32_128) { TestModuleRoutine<float>(128); }

TEST_F(TestDLLMAdamW, TestModuleF32_512) { TestModuleRoutine<float>(512); }

TEST_F(TestDLLMAdamW, TestModuleF32_1024) { TestModuleRoutine<float>(1024); }
