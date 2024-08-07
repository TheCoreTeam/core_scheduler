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
#include <torch/csrc/api/include/torch/types.h>
#include <torch/csrc/autograd/autograd.h>

#include "compute/gelu.h"
#include "compute/utils.h"
#include "memory/to_torch.h"
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

class TestCSGelu : public ::testing::Test {
 protected:
  cs::DynamicScheduler scheduler{0};

  template <typename Element>
  void TestRoutine(int T, double tol_forward, double tol_backward);
};

template <typename Element>
void TestCSGelu::TestRoutine(const int T, const double tol_forward,
                             const double tol_backward) {
  torch::manual_seed(1);
  const int B = 2;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);

  auto input2 = cs::compute::Utils::randn(scheduler, {B, T}, option);
  auto state = cs::compute::GeLU::init(scheduler);
  auto tensorOutput = cs::compute::GeLU::forward(scheduler, state, input2);
  auto GradOutput_ = cs::compute::Utils::randn_like(scheduler, tensorOutput);
  auto tensorGradInput =
      cs::compute::GeLU::backward(scheduler, state, GradOutput_);

  auto input = cs::memory::to_torch(scheduler, input2);
  input2.wait();
  auto GradOutput = cs::memory::to_torch(scheduler, GradOutput_);
  GradOutput_.wait();

  auto input1 = input.detach().clone().set_requires_grad(true);

  // 应用GELU激活函数
  auto Output1 = at::gelu(input1);

  const auto GradInput1 = torch::autograd::grad(
      {Output1}, {input1}, {GradOutput}, /*retain_graph=*/false,
      /*create_graph=*/false, /*allow_unused=*/true)[0];

  ASSERT_TRUE(at::allclose(Output1, tensorOutput));
  ASSERT_TRUE(at::allclose(GradInput1, tensorGradInput));
}

TEST_F(TestCSGelu, TestF32_128) { TestRoutine<float>(128, 5e-4, 5e-4); }

TEST_F(TestCSGelu, TestF32_512) { TestRoutine<float>(512, 5e-4, 5e-4); }

TEST_F(TestCSGelu, TestF32_1024) { TestRoutine<float>(1024, 5e-4, 5e-4); }

TEST_F(TestCSGelu, TestF16_128) { TestRoutine<nv_half>(128, 1e-2, 1e-2); }

TEST_F(TestCSGelu, TestF16_512) { TestRoutine<nv_half>(512, 1e-2, 1e-2); }

TEST_F(TestCSGelu, TestF16_1024) { TestRoutine<nv_half>(1024, 1e-2, 1e-2); }
