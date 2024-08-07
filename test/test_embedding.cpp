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
#include <ATen/ops/embedding.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/csrc/api/include/torch/types.h>

#include "compute/embedding.h"
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

class TestEmbedding : public ::testing::Test {
 protected:
  cs::DynamicScheduler scheduler{0};

  template <typename Element>
  void TestRoutine(const double tol_forward, const double tol_backward);
};

template <typename Element>
void TestEmbedding::TestRoutine(const double tol_forward,
                                const double tol_backward) {
  at::manual_seed(1);
  const int B = 2;
  const int T = 1024;
  const int MaxT = 1024;
  const int vocab = 4095;
  const int d = 512;

  at::Device device = at::kCUDA;
  torch::Dtype dtype = TypeToTorch<Element>::type;

  auto input = cs::compute::Utils::randint(
      scheduler, 0, 3, {B, T},
      torch::TensorOptions().dtype(torch::kInt).device(device));
  auto state = cs::compute::Embedding::init(
      scheduler,
      cs::compute::Embedding::Options(vocab, d).device(device).dtype(dtype));

  auto output = cs::compute::Embedding::forward(scheduler, state, input);
  output.wait();
  auto grad_output = cs::compute::Utils::randn_like(scheduler, output);
  cs::compute::Embedding::backward(scheduler, state, grad_output);
  auto input_torch = cs::memory::to_torch(scheduler, input);
  input.wait();
  auto weight_torch = cs::memory::to_torch(scheduler, state->forward.weight);
  state->forward.weight.wait();
  weight_torch.requires_grad_(true);
  const auto output_torch = at::embedding(weight_torch, input_torch);
  auto grad_output_torch = cs::memory::to_torch(scheduler, grad_output);
  grad_output.wait();
  output_torch.backward(grad_output_torch);

  ASSERT_TRUE(torch::allclose(output, output_torch));
  ASSERT_TRUE(torch::allclose(state->forward.grad_weight, weight_torch.grad()));
}

TEST_F(TestEmbedding, TestFloat) { TestRoutine<float>(1e-5, 1e-5); }

TEST_F(TestEmbedding, TestHalf) { TestRoutine<half>(1e-5, 1e-2); }
