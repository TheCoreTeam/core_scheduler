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

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <gtest/gtest.h>

#include "compute/linear.h"
#include "compute/utils.h"
#include "logger.h"
#include "memory/to_torch.h"
#include "module/linear_gelu.h"
#include "threading/dynamic_scheduler.h"

void saveTensorToFile(const torch::Tensor& tensor,
                      const std::string& filename);

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
struct TypeToTorch<nv_bfloat16> {
  using Type = c10::BFloat16;
  static const at::ScalarType type = torch::kBFloat16;
};
template <>
struct TypeToTorch<double> {
  using Type = double;
  static const at::ScalarType type = torch::kDouble;
};

class LinearGeluTestFixture : public ::testing::Test {
 protected:
  cs::DynamicScheduler scheduler_{0};
};

namespace {
template <typename Element>
void TestBackwardT(cs::Scheduler &scheduler) {
  const int m = 32, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  auto x = cs::compute::Utils::randn(scheduler, {m, s, k}, option);
  auto state = cs::compute::Linear::init(
      scheduler,
      cs::compute::Linear::Options{k, n}.bias(false).device(device).dtype(
          dtype));
  auto y = cs::compute::Linear::forward(scheduler, state, x);
  auto yGrad = cs::compute::Utils::randn_like(scheduler, y);
  auto dx = cs::compute::Linear::backwardInput(scheduler, state, yGrad);
  cs::compute::Linear::backwardParameter(scheduler, state, yGrad);
  dx.wait();
  state->forward.grad_weight.wait();
  auto xRef = cs::memory::toTorch(scheduler, x);
  x.wait();
  xRef.requires_grad_(true);
  auto wRef = cs::memory::toTorch(scheduler, state->forward.weight);
  state->forward.weight.wait();
  wRef.requires_grad_(true);
  auto yRef = torch::linear(xRef, wRef);
  auto yGradRef = cs::memory::toTorch(scheduler, yGrad);
  yGrad.wait();
  yRef.backward(yGradRef);

  ASSERT_TRUE(torch::allclose(y, yRef));
  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
  ASSERT_TRUE(torch::allclose(state->forward.grad_weight, wRef.grad()));
}
}  // namespace

//TEST_F(LinearGeluTestFixture, TestBackwardF16) {
//  TestBackwardT<nv_half>(scheduler_);
//}
//TEST_F(LinearGeluTestFixture, TestBackwardF32) { TestBackwardT<float>(scheduler_); }
//TEST_F(LinearGeluTestFixture, TestBackwardF64) {
//  TestBackwardT<double>(scheduler_);
//}

namespace {
template <typename Element>
void TestModuleT(cs::Scheduler &scheduler, bool with_bias=true) {
  const int m = 32, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  auto x = cs::compute::Utils::randn(scheduler, {m, s, k}, option);
  cs::module::LinearGelu fc{
      scheduler,
      cs::module::LinearGelu::Options{k, n}.bias(with_bias).device(device).dtype(
          dtype)};
//  fc->state()->forward.bias = cs::compute::Utils::zeros(scheduler, n, option);
//  fc->state()->forward.bias.wait();
  auto y = fc->forward(scheduler, x);
  auto yGrad = cs::compute::Utils::randn_like(scheduler, y);
//  auto dx = fc->backward(scheduler, yGrad);
//  dx.wait();
//  fc->state()->forward.grad_weight.wait();
  auto xRef = cs::memory::toTorch(scheduler, x);
  x.wait();
  xRef.requires_grad_(true);
  auto wRef = cs::memory::toTorch(scheduler, fc->state()->forward.weight);
  fc->state()->forward.weight.wait();
  wRef.requires_grad_(true);

  at::Tensor yRef;
  if(with_bias){
    auto biasRef = cs::memory::toTorch(scheduler, fc->state()->forward.bias);
    fc->state()->forward.bias.wait();
    biasRef.requires_grad_(true);
    yRef = torch::linear(xRef, wRef, biasRef);
  }else{
    yRef = torch::linear(xRef, wRef);
  }
  yRef = torch::gelu(yRef);
//  auto yGradRef = cs::memory::toTorch(scheduler, yGrad);
//  yGrad.wait();
//  yRef.backward(yGradRef);

  auto y_torch = cs::memory::toTorch(scheduler, y);
  y.wait();
  auto close = torch::allclose(y_torch, yRef, 0.0625);
  if(!close){
    saveTensorToFile(y_torch, "y.txt");
    saveTensorToFile(yRef, "yRef.txt");
    std::cout << (yRef-y_torch).abs().max() << "\n";
  }
  ASSERT_TRUE(close);
//  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
//  ASSERT_TRUE(torch::allclose(fc->state()->forward.grad_weight, wRef.grad()));
}
}  // namespace

TEST_F(LinearGeluTestFixture, TestModuleF16WithBias) { TestModuleT<nv_half>(scheduler_, true); }
TEST_F(LinearGeluTestFixture, TestModuleF16WithoutBias) { TestModuleT<nv_half>(scheduler_, false); }
TEST_F(LinearGeluTestFixture, TestModuleBF16WithBias) { TestModuleT<nv_bfloat16>(scheduler_, true); }
TEST_F(LinearGeluTestFixture, TestModuleBF16WithoutBias) { TestModuleT<nv_bfloat16>(scheduler_, false); }
