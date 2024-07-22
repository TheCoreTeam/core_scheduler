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
#include <gtest/gtest.h>
#include <torch/nn/modules/normalization.h>

#include "autocast.h"
#include "compute/embedding.h"
#include "compute/utils.h"
#include "memory/to_torch.h"
#include "module/layer_norm.h"
#include "module/linear.h"
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
struct TypeToTorch<__nv_bfloat16> {
  using Type = c10::Half;
  static const at::ScalarType type = torch::kBFloat16;
};

template <>
struct TypeToTorch<double> {
  using Type = double;
  static const at::ScalarType type = torch::kDouble;
};

class TestAutocastFixture : public ::testing::Test {
 protected:
  cs::DynamicScheduler scheduler_{0};
};

namespace {
template <typename Element>
void TestLinearT(const cs::Scheduler &scheduler) {
  const int m = 32, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);

  auto x = cs::compute::Utils::randn(scheduler, {m, s, k}, option);
  cs::module::Linear fc{
      scheduler,
      cs::module::Linear::Options{k, n}.bias(false).device(device).dtype(
          c10::kFloat)};

  cs::Tensor y;
  at::Tensor xRef, wRef, yRef;
  {
    cs::autocast::ContextGuard guard{scheduler, dtype};
    y = fc->forward(scheduler, x);
    fc->state()->forward.grad_weight.wait();
    xRef = cs::memory::to_torch(scheduler, x);
    x.wait();
    xRef.requires_grad_(true);
    wRef = cs::memory::to_torch(scheduler, fc->state()->forward.weight);
    fc->state()->forward.weight.wait();
    wRef.requires_grad_(true);
    yRef = torch::linear(xRef, wRef);
  }

  auto yGrad =
      cs::compute::Utils::randn(scheduler, y.sizes(), option.dtype(at::kFloat));
  auto dx = fc->backward(scheduler, yGrad);
  dx.wait();
  auto yGradRef = cs::memory::to_torch(scheduler, yGrad);
  yGrad.wait();
  yRef.backward(yGradRef);

  ASSERT_TRUE(torch::allclose(y, yRef));
  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
  ASSERT_TRUE(torch::allclose(fc->state()->forward.grad_weight, wRef.grad()));
}
}  // namespace

TEST_F(TestAutocastFixture, TestLinearF16) { TestLinearT<nv_half>(scheduler_); }

TEST_F(TestAutocastFixture, TestLinearBF16) {
  TestLinearT<__nv_bfloat16>(scheduler_);
}

namespace {
template <typename Element>
void TestLayerNormT(const cs::Scheduler &scheduler) {
  const int size = 128;
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<Element>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);

  cs::module::LayerNorm lnOurs{
      scheduler,
      cs::module::LayerNorm::Options{{3 * size}}.device(device).dtype(
          c10::kFloat)};
  torch::nn::LayerNorm ln{torch::nn::LayerNormOptions({3 * size})};

  cs::Tensor y;
  at::Tensor x_torch, y_torch;
  {
    cs::autocast::ContextGuard guard{scheduler, dtype};
    auto x =
        cs::compute::Utils::rand(scheduler, {size, 2 * size, 3 * size}, option);
    y = lnOurs->forward(scheduler, x);

    x_torch = cs::memory::to_torch(scheduler, x);
    x.wait();
    x_torch.set_requires_grad(true);
    ln->to(device);
    y_torch = ln->forward(x_torch);
  }

  auto dy = cs::compute::Utils::rand(scheduler, y.sizes(),
                                     y.options().dtype(c10::kFloat));
  auto dx = lnOurs->backward(scheduler, dy);
  auto dy_torch = cs::memory::to_torch(scheduler, dy);
  dy.wait();
  y_torch.backward(dy_torch);
  ASSERT_TRUE(at::allclose(y_torch, y));
  ASSERT_TRUE(
      at::allclose(ln->weight.grad(), lnOurs->state()->forward.grad_weight));
  ASSERT_TRUE(
      at::allclose(ln->bias.grad(), lnOurs->state()->forward.grad_bias));
}
}  // namespace

TEST_F(TestAutocastFixture, TestLayerNormBF16) {
  TestLayerNormT<__nv_bfloat16>(scheduler_);
}

template <typename Element>
void TestEmbeddingT(const cs::Scheduler &scheduler) {
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
      scheduler, cs::compute::Embedding::Options(vocab, d).device(device).dtype(
                     c10::kFloat));

  cs::Tensor output;
  at::Tensor output_torch, weight_torch;
  {
    cs::autocast::ContextGuard guard{scheduler, dtype};
    output = cs::compute::Embedding::forward(scheduler, state, input);
    output.wait();
    auto input_torch = cs::memory::to_torch(scheduler, input);
    input.wait();
    weight_torch = cs::memory::to_torch(scheduler, state->forward.weight);
    state->forward.weight.wait();
    weight_torch.requires_grad_(true);
    output_torch = at::embedding(weight_torch, input_torch);
  }

  auto grad_output = cs::compute::Utils::randn(
      scheduler, output.sizes(), output.options().dtype(c10::kFloat));
  cs::compute::Embedding::backward(scheduler, state, grad_output);
  auto grad_output_torch = cs::memory::to_torch(scheduler, grad_output);
  grad_output.wait();
  output_torch.backward(grad_output_torch);

  ASSERT_TRUE(torch::allclose(output, output_torch));
  ASSERT_TRUE(torch::allclose(state->forward.grad_weight, weight_torch.grad()));
}

TEST_F(TestAutocastFixture, TestEmbeddingBF16) {
  TestEmbeddingT<__nv_bfloat16>(scheduler_);
}
