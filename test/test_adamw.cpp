#include <ATen/Context.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include "compute/utils.h"
#include "memory/to_torch.h"
#include "optimizer/adamw.h"
#include "tensor.h"
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

class TestDLLMAdamW : public ::testing::Test {
 protected:
  dllm::ThreadStreamCudart stream{0};
  dllm::ThreadPoolCompute tp{0, 1};

  template <typename Element>
  void TestRoutine(int size);
};

template <typename Element>
void TestDLLMAdamW::TestRoutine(const int size) {
  const double lr = 1e-3;
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double eps = 1e-8;
  const double weight_decay = 1e-2;
  const long t = 0;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  torch::manual_seed(1);
  auto x = dllm::Tensor::create();
  {
    auto task = dllm::compute::Utils::rand(x, {size}, option);
    tp.submit(std::move(task));
  }
  std::shared_ptr<dllm::optimizer::AdamW::State> state;
  {
    auto task = dllm::optimizer::AdamW::init(state, x, lr, beta1, beta2, eps,
                                             weight_decay, false, t);
    tp.submit(std::move(task));
  }
  auto dx = dllm::Tensor::create();
  {
    auto task = dllm::compute::Utils::rand(dx, {size}, option);
    tp.submit(std::move(task));
  }
  {
    auto task = dllm::optimizer::AdamW::step(state, x, dx);
    tp.submit(std::move(task));
  }
  at::Tensor x_torch, dx_torch;
  {
    auto task = dllm::memory::toTorch(x_torch, x);
    stream.submit(std::move(task));
    x->wait();
  }
  {
    auto task = dllm::memory::toTorch(dx_torch, dx);
    stream.submit(std::move(task));
    dx->wait();
  }
  torch::manual_seed(1);
  x_torch = torch::rand_like(x_torch);
  auto m_torch = torch::zeros_like(x_torch);
  auto v_torch = torch::zeros_like(x_torch);

  x_torch = x_torch - lr * weight_decay * x_torch;
  m_torch = beta1 * m_torch + (1 - beta1) * dx_torch;
  v_torch = beta2 * v_torch + (1 - beta2) * dx_torch.square();
  auto m_hat = m_torch / (1 - std::pow(beta1, t + 1));
  auto v_hat = v_torch / (1 - std::pow(beta2, t + 1));
  x_torch = x_torch - lr * m_hat / (v_hat.sqrt() + eps);

  ASSERT_TRUE(torch::allclose(x_torch, x));
  ASSERT_TRUE(torch::allclose(m_torch, state->tensors.m));
  ASSERT_TRUE(torch::allclose(v_torch, state->tensors.v));
}

TEST_F(TestDLLMAdamW, TestF32_128) { TestRoutine<float>(128); }

TEST_F(TestDLLMAdamW, TestF32_512) { TestRoutine<float>(512); }

TEST_F(TestDLLMAdamW, TestF32_1024) { TestRoutine<float>(1024); }
