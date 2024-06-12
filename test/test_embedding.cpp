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
  dllm::DynamicScheduler scheduler{0};

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
  //  const int B = 1;
  //  const int T = 8;
  //  const int MaxT = 8;
  //  const int vocab = 16;
  //  const int d = 8;

  at::Device device = at::kCUDA;
  torch::Dtype dtype = TypeToTorch<Element>::type;

  dllm::Tensor input;
  std::shared_ptr<dllm::compute::Embedding::State> state;
  dllm::compute::Utils::randint(
      scheduler, input, 0, 3, {B, T},
      torch::TensorOptions().dtype(torch::kInt).device(device));
  dllm::compute::Embedding::init(
      scheduler, state,
      dllm::compute::Embedding::Options(vocab, d).device(device).dtype(dtype));

  dllm::Tensor output;
  dllm::compute::Embedding::forward(scheduler, state, output, input);
  output.wait();
  dllm::Tensor grad_output;
  dllm::compute::Utils::randn_like(scheduler, grad_output, output);
  dllm::compute::Embedding::backward(scheduler, state, grad_output);
  torch::Tensor input_torch;
  dllm::memory::toTorch(scheduler, input_torch, input);
  input.wait();
  torch::Tensor weight_torch;
  dllm::memory::toTorch(scheduler, weight_torch, state->forward.weight);
  state->forward.weight.wait();
  weight_torch.requires_grad_(true);
  const auto output_torch = at::embedding(weight_torch, input_torch);
  torch::Tensor grad_output_torch;
  dllm::memory::toTorch(scheduler, grad_output_torch, grad_output);
  grad_output.wait();
  output_torch.backward(grad_output_torch);

  ASSERT_TRUE(torch::allclose(output, output_torch));
  ASSERT_TRUE(torch::allclose(state->forward.grad_weight, weight_torch.grad()));
}

TEST_F(TestEmbedding, TestFloat) { TestRoutine<float>(1e-5, 1e-5); }

TEST_F(TestEmbedding, TestHalf) { TestRoutine<half>(1e-5, 1e-2); }
