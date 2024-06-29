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
#include "module/amp_linear.h"
#include "optimizer/amp_adamw.h"
#include "tensor.h"
#include "threading/dynamic_scheduler.h"

void saveTensorToFile(const torch::Tensor& tensor, const std::string& filename);

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

class TestAmpAdamW : public ::testing::Test {
 protected:
  cs::DynamicScheduler scheduler{0};

  template <typename Element>
  void TestFunctionalRoutine(int size);

  template <typename Element>
  void TestModuleRoutine(int size);
};

template <typename Element>
void TestAmpAdamW::TestModuleRoutine(const int size) {
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
  cs::module::AmpLinear fc{
      scheduler,
      cs::module::AmpLinear::Options{k, n}.bias(false).device(device).dtype(
          dtype)};
  auto y = fc->forward(scheduler, x);
  auto yGrad = cs::compute::Utils::randn_like(scheduler, y);
  auto dx = fc->backward(scheduler, yGrad);
  dx.wait();
  fc->state()->forward.grad_weight.wait();
  auto xRef = cs::memory::toTorch(scheduler, x);
  x.wait();
  xRef.requires_grad_(true);
  auto wRef = cs::memory::toTorch(scheduler, fc->state()->forward.weight);
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
  auto yGradRef = cs::memory::toTorch(scheduler, yGrad);
  yGrad.wait();
  optimTorch.zero_grad();
  yRef.backward(yGradRef);

  ASSERT_TRUE(torch::allclose(y, yRef));
  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
  ASSERT_TRUE(torch::allclose(fc->state()->forward.grad_weight,
                              fcTorch->weight.grad()));

  // optimTorch.step();
  auto wFp32 = cs::memory::toTorch(
      scheduler, std::dynamic_pointer_cast<cs::module::AmpState>(fc->state())
                     ->parametersFp32()["weight"]);
  auto m_torch = torch::zeros_like(wFp32);
  auto v_torch = torch::zeros_like(wFp32);

  wFp32 = wFp32 - lr * weight_decay * wFp32;
  m_torch =
      beta1 * m_torch + (1 - beta1) * fcTorch->weight.grad().to(at::kFloat);
  v_torch = beta2 * v_torch +
            (1 - beta2) * fcTorch->weight.grad().to(at::kFloat).square();
  auto m_hat = m_torch / (1 - std::pow(beta1, t + 1));
  auto v_hat = v_torch / (1 - std::pow(beta2, t + 1));
  wFp32 = wFp32 - lr * m_hat / (v_hat.sqrt() + eps);

  cs::optimizer::AmpAdamW::init(scheduler, fc,
                                cs::optimizer::AmpAdamW::Options{}
                                    .lr(lr)
                                    .beta1(beta1)
                                    .beta2(beta2)
                                    .eps(eps)
                                    .weight_decay(weight_decay)
                                    .t(t));
  cs::optimizer::AmpAdamW::step(scheduler, fc);

  ASSERT_TRUE(torch::allclose(fc->state()->forward.weight, wFp32.to(dtype),
                              1e-4, 1e-3));
}

TEST_F(TestAmpAdamW, TestModuleF32_128) { TestModuleRoutine<float>(128); }

TEST_F(TestAmpAdamW, TestModuleF32_512) { TestModuleRoutine<float>(512); }

TEST_F(TestAmpAdamW, TestModuleF32_1024) { TestModuleRoutine<float>(1024); }

TEST_F(TestAmpAdamW, TestModuleF16_128) { TestModuleRoutine<nv_half>(128); }

TEST_F(TestAmpAdamW, TestModuleF16_512) { TestModuleRoutine<nv_half>(512); }

TEST_F(TestAmpAdamW, TestModuleF16_1024) { TestModuleRoutine<nv_half>(1024); }
