#include <ATen/Context.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <gtest/gtest.h>
#include <torch/nn/modules/normalization.h>

#include "compute/layer_norm.h"
#include "compute/utils.h"
#include "memory/to_torch.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_cudart.h"

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

class TestLayerNormFixture : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute tp{0, 2};
  dllm::ThreadStreamCudart stream{0};

  template <typename T>
  void Test(int size);
};

template <typename T>
void TestLayerNormFixture::Test(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);

  std::shared_ptr<dllm::compute::LayerNorm::State> state;
  auto x = dllm::Tensor::create();
  auto dx = dllm::Tensor::create();
  auto y = dllm::Tensor::create();
  auto dy = dllm::Tensor::create();
  DLLM_SUBMIT_TASK(
      tp,
      dllm::compute::LayerNorm::init(
          state,
          dllm::compute::LayerNorm::Options{{3 * size}}.device(device).dtype(
              dtype)));
  DLLM_SUBMIT_TASK(
      tp, dllm::compute::Utils::rand(x, {size, 2 * size, 3 * size}, option));
  DLLM_SUBMIT_TASK(tp, dllm::compute::LayerNorm::forward(state, y, x));
  DLLM_SUBMIT_TASK(tp, dllm::compute::Utils::rand_like(dy, y));
  DLLM_SUBMIT_TASK(tp, dllm::compute::LayerNorm::backward(state, dx, dy));

  at::Tensor x_torch, dx_torch, y_ref_torch, dy_torch;
  DLLM_SUBMIT_TASK(stream, dllm::memory::toTorch(x_torch, x));
  x->wait();
  DLLM_SUBMIT_TASK(stream, dllm::memory::toTorch(y_ref_torch, y));
  y->wait();
  DLLM_SUBMIT_TASK(stream, dllm::memory::toTorch(dy_torch, dy));
  dy->wait();
  x_torch.set_requires_grad(true);
  torch::nn::LayerNorm ln{torch::nn::LayerNormOptions({3 * size})};
  ln->to(device, dtype);
  auto y_torch = ln->forward(x_torch);
  y_torch.backward(dy_torch);
  ASSERT_TRUE(at::allclose(y_torch, y));
  ASSERT_TRUE(at::allclose(x_torch.grad(), dx));
  ASSERT_TRUE(at::allclose(ln->weight.grad(), state->forward.grad_weight));
  ASSERT_TRUE(at::allclose(ln->bias.grad(), state->forward.grad_bias));
}

TEST_F(TestLayerNormFixture, TestF32) { Test<float>(128); }
TEST_F(TestLayerNormFixture, TestF64) { Test<double>(128); }
