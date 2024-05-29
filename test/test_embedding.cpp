#include <gtest/gtest.h>
#include <torch/torch.h>

#include "compute/embedding.h"
#include "compute/utils.h"
#include "memory/to_torch.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_cudart.h"

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

class TestEmbedding : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute tp{0, 2};
  dllm::ThreadStreamCudart stream{0};

  template <typename Element>
  void TestRoutine(const double tol_forward, const double tol_backward);
};

template <typename Element>
void TestEmbedding::TestRoutine(const double tol_forward,
                                const double tol_backward) {
  torch::manual_seed(1);
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

  torch::Device device = torch::kCUDA;
  torch::Dtype dtype = TypeToTorch<Element>::type;

  const auto input = dllm::Tensor::create();
  std::shared_ptr<dllm::compute::Embedding::State> state;
  {
    auto task = dllm::compute::Utils::randint(
        input, 0, 3, {B, T},
        torch::TensorOptions().dtype(torch::kInt).device(device));
    tp.submit(std::move(task));
  }
  {
    auto task = dllm::compute::Embedding::init(state, vocab, d, {}, {}, {}, {},
                                               {}, device, dtype);
    tp.submit(std::move(task));
  }

  const auto output = dllm::Tensor::create();
  {
    auto task = dllm::compute::Embedding::forward(state, output, input);
    tp.submit(std::move(task));
    output->wait();
  }
  const auto grad_output = dllm::Tensor::create();
  {
    auto task = dllm::compute::Utils::randn_like(grad_output, output);
    tp.submit(std::move(task));
  }
  {
    auto task = dllm::compute::Embedding::backward(state, grad_output);
    tp.submit(std::move(task));
  }

  torch::Tensor input_torch;
  {
    auto task = dllm::memory::toTorch(input_torch, input);
    stream.submit(std::move(task));
    input->wait();
  }
  torch::Tensor weight_torch;
  {
    auto task = dllm::memory::toTorch(weight_torch, state->forward.weight);
    stream.submit(std::move(task));
    state->forward.weight->wait();
  }
  weight_torch.requires_grad_(true);
  auto output_torch = at::embedding(weight_torch, input_torch);
  torch::Tensor grad_output_torch;
  {
    auto task = dllm::memory::toTorch(grad_output_torch, grad_output);
    stream.submit(std::move(task));
    grad_output->wait();
  }
  output_torch.backward(grad_output_torch);

  ASSERT_TRUE(torch::allclose(output, output_torch));
  ASSERT_TRUE(torch::allclose(state->forward.grad_weight, weight_torch.grad()));
}

TEST_F(TestEmbedding, TestFloat) { TestRoutine<float>(1e-5, 1e-5); }

TEST_F(TestEmbedding, TestHalf) { TestRoutine<half>(1e-5, 1e-2); }