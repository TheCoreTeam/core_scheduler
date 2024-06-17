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

#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/nn/functional.h>

#include "compute/scaled_dot_product_attention.h"
#include "compute/utils.h"
#include "memory/to_torch.h"
#include "threading/dynamic_scheduler.h"

class FlashAttentionTestFixture : public ::testing::Test {
 protected:
  cs::DynamicScheduler scheduler{0};
};

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

namespace {
template <typename Element>
void TestT(cs::Scheduler& scheduler) {
  static_assert(std::is_same_v<Element, nv_half> ||
                std::is_same_v<Element, nv_bfloat16>);
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  torch::manual_seed(1);
  int B = 15, T = 7, n_head = 8, n_embd = n_head * 8;
  auto qkv = cs::compute::Utils::rand(scheduler, {B, T, n_embd * 3}, option);
  auto qkvSplit = cs::compute::Utils::split(scheduler, qkv, n_embd, -1);
  auto& q = qkvSplit[0];
  auto& k = qkvSplit[1];
  auto& v = qkvSplit[2];

  auto scale = 1.0 / std::sqrt(k.size(-1));
  auto state = cs::compute::ScaledDotProductFlashAttention::init(
      scheduler, cs::compute::ScaledDotProductFlashAttention::Options{}
                     .is_causal(true)
                     .scale(scale));
  auto qview =
      cs::compute::Utils::view(scheduler, q, {B, T, n_head, n_embd / n_head});
  qview.wait();
  auto kview =
      cs::compute::Utils::view(scheduler, k, {B, T, n_head, n_embd / n_head});
  kview.wait();
  auto vview =
      cs::compute::Utils::view(scheduler, v, {B, T, n_head, n_embd / n_head});
  vview.wait();
  auto output = cs::compute::ScaledDotProductFlashAttention::forward(
      scheduler, state, qview, kview, vview);
  output.wait();
  auto dout = cs::compute::Utils::rand_like(scheduler, output);
  output.wait();
  auto [dq, dk, dv] = cs::compute::ScaledDotProductFlashAttention::backward(
      scheduler, state, dout);
  dq.wait();

  auto qkv_torch = cs::memory::toTorch(scheduler, qkv);
  qkv.wait();
  auto dout_torch = cs::memory::toTorch(scheduler, dout);
  dout.wait();

  auto qkv_torch_v = qkv_torch.split(n_embd, -1);
  auto q_torch = qkv_torch_v[0]
                     .view({B, T, n_head, n_embd / n_head})
                     .detach()
                     .requires_grad_(true);
  auto k_torch = qkv_torch_v[1]
                     .view({B, T, n_head, n_embd / n_head})
                     .detach()
                     .requires_grad_(true);
  auto v_torch = qkv_torch_v[2]
                     .view({B, T, n_head, n_embd / n_head})
                     .detach()
                     .requires_grad_(true);

  // Transpose for attention calculation
  auto qT = q_torch.transpose(1, 2);
  auto kT = k_torch.transpose(1, 2);
  auto vT = v_torch.transpose(1, 2);

  // Create lower triangular mask
  auto mask = torch::tril(torch::ones({T, T}, option)).view({1, 1, T, T});

  auto att = torch::matmul(qT, kT.transpose(-2, -1)) * scale;
  att = att.masked_fill(mask == 0, -std::numeric_limits<float>::infinity());
  att = torch::softmax(att, -1);
  att = torch::dropout(att, /*p=*/0.0, /*train=*/false);

  // Apply attention to values
  auto y_attn = torch::matmul(att, vT);
  y_attn = y_attn.transpose(1, 2).contiguous();
  y_attn.backward(dout_torch);

  ASSERT_TRUE(torch::allclose(y_attn, output, 1e-5, 2e-2));

  ASSERT_TRUE(torch::allclose(dq, q_torch.grad(), 1e-5, 2e-2));
  ASSERT_TRUE(torch::allclose(dk, k_torch.grad(), 1e-5, 2e-2));
  ASSERT_TRUE(torch::allclose(dv, v_torch.grad(), 1e-5, 2e-2));
}
}  // namespace

TEST_F(FlashAttentionTestFixture, TestF16) { TestT<nv_half>(scheduler); }
TEST_F(FlashAttentionTestFixture, TestBF16) { TestT<nv_bfloat16>(scheduler); }
