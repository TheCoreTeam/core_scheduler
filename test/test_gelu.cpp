#include <gtest/gtest.h>
#include <torch/torch.h>

#include "compute/gelu.h"
#include "compute/utils.h"
#include "logger.h"
#include "memory/to_torch.h"
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

class TestDLLMGelu : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute tp{0, 2};
  dllm::ThreadStreamCudart stream{0};

  template <typename Element>
  void TestRoutine(int T, double tol_forward, double tol_backward);
};

template <typename Element>
void TestDLLMGelu::TestRoutine(const int T, const double tol_forward,
                               const double tol_backward) {
  torch::manual_seed(1);
  const int B = 2;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);

  const auto input2 = dllm::Tensor::create();
  const auto tensorGradInput = dllm::Tensor::create();
  const auto tensorOutput = dllm::Tensor::create();
  const auto GradOutput_ = dllm::Tensor::create();
  std::shared_ptr<dllm::compute::GeLU::State> state;
  {
    auto task = dllm::compute::Utils::randn(input2, {B, T}, option);
    tp.submit(std::move(task));
  }
  {
    auto task = dllm::compute::GeLU::init(state);
    tp.submit(std::move(task));
  }
  {
    auto task = dllm::compute::GeLU::forward(state, tensorOutput, input2);
    tp.submit(std::move(task));
  }
  {
    auto task = dllm::compute::Utils::randn_like(GradOutput_, tensorOutput);
    tp.submit(std::move(task));
  }
  {
    auto task =
        dllm::compute::GeLU::backward(state, tensorGradInput, GradOutput_);
    tp.submit(std::move(task));
  }

  torch::Tensor input;
  torch::Tensor GradOutput;
  {
    auto task = dllm::memory::toTorch(input, input2);
    stream.submit(std::move(task));
    input2->wait();
  }
  {
    auto task = dllm::memory::toTorch(GradOutput, GradOutput_);
    stream.submit(std::move(task));
    GradOutput_->wait();
  }

  auto input1 = input.detach().clone().set_requires_grad(true);

  // 应用GELU激活函数
  auto Output1 = torch::gelu(input1);

  const auto GradInput1 = torch::autograd::grad(
      {Output1}, {input1}, {GradOutput}, /*retain_graph=*/false,
      /*create_graph=*/false, /*allow_unused=*/true)[0];

  ASSERT_TRUE(at::allclose(Output1, tensorOutput));
  ASSERT_TRUE(at::allclose(GradInput1, tensorGradInput));
}

TEST_F(TestDLLMGelu, TestF32_128) { TestRoutine<float>(128, 5e-4, 5e-4); }

TEST_F(TestDLLMGelu, TestF32_512) { TestRoutine<float>(512, 5e-4, 5e-4); }

TEST_F(TestDLLMGelu, TestF32_1024) { TestRoutine<float>(1024, 5e-4, 5e-4); }

TEST_F(TestDLLMGelu, TestF16_128) { TestRoutine<nv_half>(128, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF16_512) { TestRoutine<nv_half>(512, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF16_1024) { TestRoutine<nv_half>(1024, 1e-2, 1e-2); }

// TEST_F(TestDLLMGelu, TestF64_128) { TestRoutine<double>(128, 1e-10, 1e-10); }

// TEST_F(TestDLLMGelu, TestF64_512) { TestRoutine<double>(512, 1e-10, 1e-10); }

// TEST_F(TestDLLMGelu, TestF64_1024) { TestRoutine<double>(1024, 1e-10, 1e-10);
// }
