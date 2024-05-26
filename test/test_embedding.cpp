#include <gtest/gtest.h>
#include <torch/torch.h>

#include "compute/embedding.h"
#include "threading/thread_pool_compute.h"

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
  auto wte = torch::randn({vocab, d},
                          torch::TensorOptions().dtype(dtype).device(device));
  auto input = torch::randint(
      0, 3, {B, T}, torch::TensorOptions().dtype(torch::kInt).device(device));
  auto input_wpe =
      torch::range(0, T - 1,
                   torch::TensorOptions().dtype(torch::kInt).device(device))
          .repeat({B, 1});

  auto input1 = input.detach().clone();
  auto wte1 = wte.to(TypeToTorch<Element>::type)
                  .detach()
                  .clone()
                  .set_requires_grad(true);

  auto input2 = input.detach().clone();
  auto wte2 = wte.detach().clone();

  auto output1 = at::embedding(wte1, input1);

  auto state = dllm::compute::Embedding::init(vocab, d, {}, {}, {}, {}, {},
                                              device, dtype);
  state->forward.weight->tensor().copy_(wte1.detach());

  const auto output2 = std::make_shared<dllm::Tensor>();
  {
    auto task = dllm::compute::Embedding::forward(
        state, output2, std::make_shared<dllm::Tensor>(input));
    tp.submit(std::move(task));
    output2->wait();
  }

  ASSERT_TRUE(torch::allclose(output1, output2->tensor()));
  // backward check
  auto grad_output = torch::rand_like(output1);

  // 计算梯度
  auto grads = torch::autograd::grad(
      {output1}, {wte1}, {grad_output}, /*retain_graph=*/false,
      /*create_graph=*/false, /*allow_unused=*/true);

  // Access and print gradients
  auto grad_wte = grads[0];

  const auto grad2 = std::make_shared<dllm::Tensor>();
  {
    auto task = dllm::compute::Embedding::backward(
        state, grad2, std::make_shared<dllm::Tensor>(grad_output));
    tp.submit(std::move(task));
    grad2->wait();
  }
  ASSERT_TRUE(torch::allclose(grad_wte, grad2->tensor()));
}

TEST_F(TestEmbedding, TestFloat) { TestRoutine<float>(1e-5, 1e-5); }

TEST_F(TestEmbedding, TestHalf) { TestRoutine<half>(1e-5, 1e-2);
}