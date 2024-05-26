#include <gtest/gtest.h>
#include <torch/torch.h>

#include "compute/linear.h"
#include "logger.h"
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

class LinearThreadPoolComputeTestFixture : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute threadPool{0, 1};
};

namespace {
template <typename Element>
void TestThreadPoolComputeForwardT(dllm::ThreadPoolCompute &threadPool) {
  const int m = 128, n = 2048, k = 512, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  auto x = std::make_shared<dllm::Tensor>(at::randn({m, s, k}, option));
  auto w = std::make_shared<dllm::Tensor>(at::randn({n, k}, option));
  auto y = std::make_shared<dllm::Tensor>();

  {
    auto task = dllm::compute::Linear::forward(y, x, w);
    threadPool.submit(std::move(task));
    y->wait();
  }

  ASSERT_TRUE(
      torch::allclose(y->tensor(), torch::linear(x->tensor(), w->tensor())));
}
}  // namespace

TEST_F(LinearThreadPoolComputeTestFixture, TestForwardF16) {
  TestThreadPoolComputeForwardT<nv_half>(threadPool);
}
TEST_F(LinearThreadPoolComputeTestFixture, TestForwardF32) {
  TestThreadPoolComputeForwardT<float>(threadPool);
}
TEST_F(LinearThreadPoolComputeTestFixture, TestForwardF64) {
  TestThreadPoolComputeForwardT<double>(threadPool);
}

namespace {
template <typename Element>
void TestThreadPoolComputeBackwardT(dllm::ThreadPoolCompute &threadPool) {
  const int m = 32, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  auto x = std::make_shared<dllm::Tensor>(at::randn({m, s, k}, option));
  auto w = std::make_shared<dllm::Tensor>(at::randn({n, k}, option));
  auto y = std::make_shared<dllm::Tensor>();
  auto dx = std::make_shared<dllm::Tensor>();
  auto dw = std::make_shared<dllm::Tensor>(torch::zeros_like(w->tensor()));

  auto xRef = x->tensor().clone().requires_grad_(true);
  auto wRef = w->tensor().clone().requires_grad_(true);
  auto yRef = torch::linear(xRef, wRef);
  auto yGrad = torch::randn_like(yRef);
  yRef.backward(yGrad);

  {
    auto task = dllm::compute::Linear::forward(y, x, w);
    threadPool.submit(std::move(task));
  }
  {
    auto task = dllm::compute::Linear::backwardInput(
        dx, std::make_shared<dllm::Tensor>(yGrad), w);
    threadPool.submit(std::move(task));
  }
  {
    auto task = dllm::compute::Linear::backwardWeight(
        dw, std::make_shared<dllm::Tensor>(yGrad), x);
    threadPool.submit(std::move(task));
  }
  dx->wait();
  dw->wait();

  ASSERT_TRUE(torch::allclose(dx->tensor(), xRef.grad()));
  ASSERT_TRUE(torch::allclose(dw->tensor(), wRef.grad()));
}
}  // namespace

TEST_F(LinearThreadPoolComputeTestFixture, TestBackwardF16) {
  TestThreadPoolComputeBackwardT<nv_half>(threadPool);
}
TEST_F(LinearThreadPoolComputeTestFixture, TestBackwardF32) {
  TestThreadPoolComputeBackwardT<float>(threadPool);
}
TEST_F(LinearThreadPoolComputeTestFixture, TestBackwardF64) {
  TestThreadPoolComputeBackwardT<double>(threadPool);
}
