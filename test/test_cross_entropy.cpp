#include <ATen/Context.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <gtest/gtest.h>

#include "compute/cross_entropy.h"
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

class TestCrossEntropyFixture : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute tp{0, 2};
  dllm::ThreadStreamCudart stream{0};

  template <typename T>
  void Test(int size);
};

template <typename T>
void TestCrossEntropyFixture::Test(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  auto loss = dllm::Tensor::create();
  auto x = dllm::Tensor::create();
  auto dx = dllm::Tensor::create();
  auto target = dllm::Tensor::create();
  DLLM_SUBMIT_TASK(
      tp, dllm::compute::Utils::rand(x, {size, 2 * size, 3 * size}, option));
  DLLM_SUBMIT_TASK(tp, dllm::compute::Utils::view(x, x, {-1, x->size(-1)}));
  DLLM_SUBMIT_TASK(
      tp, dllm::compute::Utils::randint(target, 0, 3 * size, {size, 2 * size},
                                        option.dtype(at::kLong)));
  DLLM_SUBMIT_TASK(tp, dllm::compute::Utils::view(target, target, {-1}));
  std::shared_ptr<dllm::compute::CrossEntropy::State> state;
  DLLM_SUBMIT_TASK(tp, dllm::compute::CrossEntropy::init(state));
  DLLM_SUBMIT_TASK(
      tp, dllm::compute::CrossEntropy::forward(state, loss, x, target));
  DLLM_SUBMIT_TASK(tp, dllm::compute::CrossEntropy::backward(state, dx));
  at::Tensor loss_ref_torch, x_torch, dx_torch, target_torch;
  DLLM_SUBMIT_TASK(stream, dllm::memory::toTorch(loss_ref_torch, loss));
  loss->wait();
  DLLM_SUBMIT_TASK(stream, dllm::memory::toTorch(x_torch, x));
  x->wait();
  DLLM_SUBMIT_TASK(stream, dllm::memory::toTorch(dx_torch, dx));
  dx->wait();
  DLLM_SUBMIT_TASK(stream, dllm::memory::toTorch(target_torch, target));
  target->wait();
  x_torch.set_requires_grad(true);
  auto loss_torch = at::cross_entropy_loss(x_torch, target_torch);
  loss_torch.backward();
  ASSERT_TRUE(at::allclose(loss_torch, loss_ref_torch));
  ASSERT_TRUE(at::allclose(x_torch.grad(), dx));
}

TEST_F(TestCrossEntropyFixture, TestCatF32) { Test<float>(128); }
TEST_F(TestCrossEntropyFixture, TestCatF64) { Test<double>(128); }
