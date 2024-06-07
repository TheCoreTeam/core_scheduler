#include <gtest/gtest.h>
#include <torch/torch.h>

#include "compute/linear.h"
#include "compute/utils.h"
#include "logger.h"
#include "memory/to_torch.h"
#include "module/linear.h"
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

class LinearTestFixture : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute threadPool{0, 1};
};

namespace {
template <typename Element>
void TestBackwardT(dllm::ThreadPoolCompute &tp) {
  dllm::ThreadStreamCudart stream{0};
  const int m = 32, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  auto y = dllm::Tensor::create();
  auto dx = dllm::Tensor::create();
  auto x = dllm::Tensor::create();
  dllm::compute::Utils::randn(tp, x, {m, s, k}, option);
  std::shared_ptr<dllm::compute::Linear::State> state;
  dllm::compute::Linear::init(
      tp, state,
      dllm::compute::Linear::Options{k, n}.bias(false).device(device).dtype(
          dtype));
  dllm::compute::Linear::forward(tp, state, y, x);
  auto yGrad = dllm::Tensor::create();
  dllm::compute::Utils::randn_like(tp, yGrad, y);
  dllm::compute::Linear::backwardInput(tp, state, dx, yGrad);
  dllm::compute::Linear::backwardParameter(tp, state, yGrad);
  dx->wait();
  state->forward.grad_weight->wait();
  torch::Tensor xRef;
  dllm::memory::toTorch(stream, xRef, x);
  x->wait();
  xRef.requires_grad_(true);
  torch::Tensor wRef;
  dllm::memory::toTorch(stream, wRef, state->forward.weight);
  state->forward.weight->wait();
  wRef.requires_grad_(true);
  auto yRef = torch::linear(xRef, wRef);
  torch::Tensor yGradRef;
  dllm::memory::toTorch(stream, yGradRef, yGrad);
  yGrad->wait();
  yRef.backward(yGradRef);

  ASSERT_TRUE(torch::allclose(y, yRef));
  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
  ASSERT_TRUE(torch::allclose(state->forward.grad_weight, wRef.grad()));
}
}  // namespace

TEST_F(LinearTestFixture, TestBackwardF16) {
  TestBackwardT<nv_half>(threadPool);
}
TEST_F(LinearTestFixture, TestBackwardF32) { TestBackwardT<float>(threadPool); }
TEST_F(LinearTestFixture, TestBackwardF64) {
  TestBackwardT<double>(threadPool);
}

namespace {
template <typename Element>
void TestModuleT(dllm::ThreadPoolCompute &tp) {
  dllm::ThreadStreamCudart stream{0};
  const int m = 32, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  auto y = dllm::Tensor::create();
  auto dx = dllm::Tensor::create();
  auto x = dllm::Tensor::create();
  dllm::compute::Utils::randn(tp, x, {m, s, k}, option);
  dllm::module::Linear fc{
      tp, dllm::module::Linear::Options{k, n}.bias(false).device(device).dtype(
              dtype)};
  fc->forward(tp, y, x);
  auto yGrad = dllm::Tensor::create();
  dllm::compute::Utils::randn_like(tp, yGrad, y);
  fc->backward(tp, dx, yGrad);
  dx->wait();
  fc->state()->forward.grad_weight->wait();
  torch::Tensor xRef;
  dllm::memory::toTorch(stream, xRef, x);
  x->wait();
  xRef.requires_grad_(true);
  torch::Tensor wRef;
  dllm::memory::toTorch(stream, wRef, fc->state()->forward.weight);
  fc->state()->forward.weight->wait();
  wRef.requires_grad_(true);
  auto yRef = torch::linear(xRef, wRef);
  torch::Tensor yGradRef;
  dllm::memory::toTorch(stream, yGradRef, yGrad);
  yGrad->wait();
  yRef.backward(yGradRef);

  ASSERT_TRUE(torch::allclose(y, yRef));
  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
  ASSERT_TRUE(torch::allclose(fc->state()->forward.grad_weight, wRef.grad()));
}
}  // namespace

TEST_F(LinearTestFixture, TestModuleF16) { TestModuleT<nv_half>(threadPool); }
TEST_F(LinearTestFixture, TestModuleF32) { TestModuleT<float>(threadPool); }
TEST_F(LinearTestFixture, TestModuleF64) { TestModuleT<double>(threadPool); }
