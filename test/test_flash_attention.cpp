#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/nn/functional.h>

#include "compute/scaled_dot_product_attention.h"
#include "compute/utils.h"
#include "memory/to_torch.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_cudart.h"

class FlashAttentionTestFixture : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute tp{0, 2};
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
void TestT(dllm::ThreadPoolCompute& tp) {
  dllm::ThreadStreamCudart stream{0};
  static_assert(std::is_same_v<Element, nv_half> ||
                std::is_same_v<Element, nv_bfloat16>);
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  torch::manual_seed(1);
  int B = 15, T = 7, n_head = 8, n_embd = n_head * 8;
  dllm::Tensor qkv;
  dllm::compute::Utils::rand(tp, qkv, {B, T, n_embd * 3}, option);
  std::vector<dllm::ReadOnlyTensor> qkvSplit;
  dllm::compute::Utils::split(tp, qkvSplit, qkv, n_embd, -1);
  auto& q = qkvSplit[0];
  auto& k = qkvSplit[1];
  auto& v = qkvSplit[2];

  auto scale = 1.0 / std::sqrt(k.size(-1));
  auto state =
      std::make_shared<dllm::compute::ScaledDotProductFlashAttention::State>();
  dllm::compute::ScaledDotProductFlashAttention::init(
      tp, state,
      dllm::compute::ScaledDotProductFlashAttention::Options{}
          .is_causal(true)
          .scale(scale));
  dllm::Tensor qview;
  dllm::compute::Utils::view(tp, qview, q, {B, T, n_head, n_embd / n_head});
  qview.wait();
  dllm::Tensor kview;
  dllm::compute::Utils::view(tp, kview, k, {B, T, n_head, n_embd / n_head});
  kview.wait();
  dllm::Tensor vview;
  dllm::compute::Utils::view(tp, vview, v, {B, T, n_head, n_embd / n_head});
  vview.wait();
  dllm::Tensor output;
  dllm::compute::ScaledDotProductFlashAttention::forward(tp, state, output,
                                                         qview, kview, vview);
  output.wait();
  dllm::Tensor dout;
  dllm::Tensor dq;
  dllm::Tensor dk;
  dllm::Tensor dv;
  dllm::compute::Utils::rand_like(tp, dout, output);
  output.wait();
  dllm::compute::ScaledDotProductFlashAttention::backward(tp, state, dq, dk, dv,
                                                          dout);
  dq.wait();

  torch::Tensor qkv_torch;
  dllm::memory::toTorch(stream, qkv_torch, qkv);
  qkv.wait();
  torch::Tensor dout_torch;
  dllm::memory::toTorch(stream, dout_torch, dout);
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

TEST_F(FlashAttentionTestFixture, TestF16) { TestT<nv_half>(tp); }
TEST_F(FlashAttentionTestFixture, TestBF16) { TestT<nv_bfloat16>(tp); }
