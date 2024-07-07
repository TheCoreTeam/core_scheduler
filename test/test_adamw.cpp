/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/Context.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include "compute/utils.h"
#include "memory/to_torch.h"
#include "module/adamw.h"
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

class TestAdamW : public ::testing::Test {
 protected:
  cs::DynamicScheduler scheduler{0};

  template <typename Element>
  void TestFunctionalRoutine(int size);

  template <typename Element>
  void TestModuleRoutine(int size);
};

template <typename Element>
void TestAdamW::TestFunctionalRoutine(const int size) {
  const double lr = 1;
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double eps = 1e-8;
  const double weight_decay = 1e-2;
  const long t = 0;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  torch::manual_seed(1);
  auto x = cs::compute::Utils::rand(scheduler, {size}, option);
  auto state = cs::optimizer::AdamW::init(scheduler, x,
                                          cs::optimizer::AdamW::Options{}
                                              .lr(lr)
                                              .beta1(beta1)
                                              .beta2(beta2)
                                              .eps(eps)
                                              .weight_decay(weight_decay)
                                              .t(t));
  auto dx = cs::compute::Utils::rand(scheduler, {size}, option);
  cs::optimizer::AdamW::step(scheduler, state, x, dx);
  auto x_torch = cs::memory::to_torch(scheduler, x);
  x.wait();
  auto dx_torch = cs::memory::to_torch(scheduler, dx);
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

  ASSERT_TRUE(torch::allclose(x_torch, x, 1e-4, 1e-5));
  ASSERT_TRUE(torch::allclose(m_torch, state->tensors.m, 1e-4, 1e-5));
  ASSERT_TRUE(torch::allclose(v_torch, state->tensors.v, 1e-4, 1e-5));
}

TEST_F(TestAdamW, TestFunctionalF32_128) { TestFunctionalRoutine<float>(128); }

TEST_F(TestAdamW, TestFunctionalF32_512) { TestFunctionalRoutine<float>(512); }

TEST_F(TestAdamW, TestFunctionalF32_1024) {
  TestFunctionalRoutine<float>(1024);
}

template <typename Element>
void TestAdamW::TestModuleRoutine(const int size) {
  const double lr = 1;
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double eps = 1e-8;
  const double weight_decay = 1e-2;
  const long t = 0;
  const int m = size, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  auto x = cs::compute::Utils::randn(scheduler, {m, s, k}, option);
  cs::module::Linear fc{
      scheduler,
      cs::module::Linear::Options{k, n}.bias(false).device(device).dtype(
          dtype)};
  auto y = fc->forward(scheduler, x);
  auto yGrad = cs::compute::Utils::randn_like(scheduler, y);
  auto dx = fc->backward(scheduler, yGrad);
  dx.wait();
  fc->state()->forward.grad_weight.wait();
  auto xRef = cs::memory::to_torch(scheduler, x);
  x.wait();
  xRef.requires_grad_(true);
  auto wRef = cs::memory::to_torch(scheduler, fc->state()->forward.weight);
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
  auto yGradRef = cs::memory::to_torch(scheduler, yGrad);
  yGrad.wait();
  optimTorch.zero_grad();
  yRef.backward(yGradRef);

  ASSERT_TRUE(torch::allclose(y, yRef));
  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
  ASSERT_TRUE(torch::allclose(fc->state()->forward.grad_weight,
                              fcTorch->weight.grad()));

  optimTorch.step();

  cs::module::AdamW adamw{scheduler, fc,
                          cs::optimizer::AdamW::Options{}
                              .lr(lr)
                              .beta1(beta1)
                              .beta2(beta2)
                              .eps(eps)
                              .weight_decay(weight_decay)
                              .t(t)};

  adamw->step(scheduler);

  ASSERT_TRUE(torch::allclose(fc->state()->forward.weight, fcTorch->weight,
                              1e-4, 1e-5));
}

TEST_F(TestAdamW, TestModuleF32_128) { TestModuleRoutine<float>(128); }

TEST_F(TestAdamW, TestModuleF32_512) { TestModuleRoutine<float>(512); }

TEST_F(TestAdamW, TestModuleF32_1024) { TestModuleRoutine<float>(1024); }
