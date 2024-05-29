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
  auto qkv = dllm::Tensor::create();
  {
    auto task = dllm::compute::Utils::rand(qkv, {B, T, n_embd * 3}, option);
    tp.submit(std::move(task));
  }
  std::vector<std::shared_ptr<const dllm::ReadOnlyTensor>> qkvSplit;
  {
    auto task = dllm::compute::Utils::split(qkvSplit, qkv, n_embd, -1);
    tp.submit(std::move(task));
  }
  auto& q = qkvSplit[0];
  auto& k = qkvSplit[1];
  auto& v = qkvSplit[2];

  auto scale = 1.0 / std::sqrt(k->size(-1));
  auto state =
      std::make_shared<dllm::compute::ScaledDotProductFlashAttention::State>();
  {
    auto task = dllm::compute::ScaledDotProductFlashAttention::init(
        state, {}, true, {}, scale);
    tp.submit(std::move(task));
  }
  auto qview = dllm::ReadOnlyTensor::create();
  {
    auto task =
        dllm::compute::Utils::view(qview, q, {B, T, n_head, n_embd / n_head});
    tp.submit(std::move(task));
    qview->wait();
  }
  auto kview = dllm::ReadOnlyTensor::create();
  {
    auto task =
        dllm::compute::Utils::view(kview, k, {B, T, n_head, n_embd / n_head});
    tp.submit(std::move(task));
    kview->wait();
  }
  auto vview = dllm::ReadOnlyTensor::create();
  {
    auto task =
        dllm::compute::Utils::view(vview, v, {B, T, n_head, n_embd / n_head});
    tp.submit(std::move(task));
    vview->wait();
  }
  auto output = dllm::Tensor::create();
  {
    auto task = dllm::compute::ScaledDotProductFlashAttention::forward(
        state, output, qview, kview, vview);
    tp.submit(std::move(task));
    output->wait();
  }
  auto dout = dllm::Tensor::create();
  std::shared_ptr<const dllm::ReadOnlyTensor> dq, dk, dv;
  {
    auto task = dllm::compute::Utils::rand_like(dout, output);
    tp.submit(std::move(task));
    output->wait();
  }
  {
    auto task = dllm::compute::ScaledDotProductFlashAttention::backward(
        state, dq, dk, dv, dout);
    tp.submit(std::move(task));
    dq->wait();
  }

  torch::Tensor qkv_torch;
  {
    auto task = dllm::memory::toTorch(qkv_torch, qkv);
    stream.submit(std::move(task));
    qkv->wait();
  }
  torch::Tensor dout_torch;
  {
    auto task = dllm::memory::toTorch(dout_torch, dout);
    stream.submit(std::move(task));
    dout->wait();
  }

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
