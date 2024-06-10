#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "compute/linear.h"
#include "compute/utils.h"
#include "logger.h"
#include "memory/to_torch.h"
#include "module/linear.h"
#include "threading/dynamic_scheduler.h"

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
  dllm::DynamicScheduler scheduler_{0};
};

namespace {
template <typename Element>
void TestBackwardT(dllm::Scheduler &scheduler) {
  const int m = 32, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  dllm::Tensor y;
  dllm::Tensor dx;
  dllm::Tensor x;
  dllm::compute::Utils::randn(scheduler, x, {m, s, k}, option);
  std::shared_ptr<dllm::compute::Linear::State> state;
  dllm::compute::Linear::init(
      scheduler, state,
      dllm::compute::Linear::Options{k, n}.bias(false).device(device).dtype(
          dtype));
  dllm::compute::Linear::forward(scheduler, state, y, x);
  dllm::Tensor yGrad;
  dllm::compute::Utils::randn_like(scheduler, yGrad, y);
  dllm::compute::Linear::backwardInput(scheduler, state, dx, yGrad);
  dllm::compute::Linear::backwardParameter(scheduler, state, yGrad);
  dx.wait();
  state->forward.grad_weight.wait();
  torch::Tensor xRef;
  dllm::memory::toTorch(scheduler, xRef, x);
  x.wait();
  xRef.requires_grad_(true);
  torch::Tensor wRef;
  dllm::memory::toTorch(scheduler, wRef, state->forward.weight);
  state->forward.weight.wait();
  wRef.requires_grad_(true);
  auto yRef = torch::linear(xRef, wRef);
  torch::Tensor yGradRef;
  dllm::memory::toTorch(scheduler, yGradRef, yGrad);
  yGrad.wait();
  yRef.backward(yGradRef);

  ASSERT_TRUE(torch::allclose(y, yRef));
  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
  ASSERT_TRUE(torch::allclose(state->forward.grad_weight, wRef.grad()));
}
}  // namespace

TEST_F(LinearTestFixture, TestBackwardF16) {
  TestBackwardT<nv_half>(scheduler_);
}
TEST_F(LinearTestFixture, TestBackwardF32) { TestBackwardT<float>(scheduler_); }
TEST_F(LinearTestFixture, TestBackwardF64) {
  TestBackwardT<double>(scheduler_);
}

namespace {
template <typename Element>
void TestModuleT(dllm::Scheduler &scheduler) {
  const int m = 32, n = 16, k = 4, s = 3;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);
  dllm::Tensor y;
  dllm::Tensor dx;
  dllm::Tensor x;
  dllm::compute::Utils::randn(scheduler, x, {m, s, k}, option);
  dllm::module::Linear fc{
      scheduler,
      dllm::module::Linear::Options{k, n}.bias(false).device(device).dtype(
          dtype)};
  fc->forward(scheduler, y, x);
  dllm::Tensor yGrad;
  dllm::compute::Utils::randn_like(scheduler, yGrad, y);
  fc->backward(scheduler, dx, yGrad);
  dx.wait();
  fc->state()->forward.grad_weight.wait();
  torch::Tensor xRef;
  dllm::memory::toTorch(scheduler, xRef, x);
  x.wait();
  xRef.requires_grad_(true);
  torch::Tensor wRef;
  dllm::memory::toTorch(scheduler, wRef, fc->state()->forward.weight);
  fc->state()->forward.weight.wait();
  wRef.requires_grad_(true);
  auto yRef = torch::linear(xRef, wRef);
  torch::Tensor yGradRef;
  dllm::memory::toTorch(scheduler, yGradRef, yGrad);
  yGrad.wait();
  yRef.backward(yGradRef);

  ASSERT_TRUE(torch::allclose(y, yRef));
  ASSERT_TRUE(torch::allclose(dx, xRef.grad()));
  ASSERT_TRUE(torch::allclose(fc->state()->forward.grad_weight, wRef.grad()));
}
}  // namespace

TEST_F(LinearTestFixture, TestModuleF16) { TestModuleT<nv_half>(scheduler_); }
TEST_F(LinearTestFixture, TestModuleF32) { TestModuleT<float>(scheduler_); }
TEST_F(LinearTestFixture, TestModuleF64) { TestModuleT<double>(scheduler_); }
