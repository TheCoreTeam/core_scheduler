/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/Context.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "autocast.h"
#include "compute/cross_entropy.h"
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
struct TypeToTorch<__nv_bfloat16> {
  using Type = c10::BFloat16;
  static const at::ScalarType type = at::kBFloat16;
};

template <>
struct TypeToTorch<double> {
  using Type = double;
  static const at::ScalarType type = at::kDouble;
};

class TestCrossEntropyFixture : public ::testing::Test {
 protected:
  cs::DynamicScheduler scheduler{0};

  template <typename T>
  void Test(int size);

  template <typename T>
  void TestAmp(int size);
};

template <typename T>
void TestCrossEntropyFixture::Test(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  auto x =
      cs::compute::Utils::rand(scheduler, {size, 2 * size, 3 * size}, option);
  x = cs::compute::Utils::view(scheduler, x, {-1, x.size(-1)});
  auto target = cs::compute::Utils::randint(
      scheduler, 0, 3 * size, {size, 2 * size}, option.dtype(at::kLong));
  target = cs::compute::Utils::view(scheduler, target, {-1});
  auto state = cs::compute::CrossEntropy::init(scheduler);
  auto loss = cs::compute::CrossEntropy::forward(scheduler, state, x, target);
  auto dx = cs::compute::CrossEntropy::backward(scheduler, state);
  auto loss_ref_torch = cs::memory::to_torch(scheduler, loss);
  loss.wait();
  auto x_torch = cs::memory::to_torch(scheduler, x);
  x.wait();
  auto dx_torch = cs::memory::to_torch(scheduler, dx);
  dx.wait();
  auto target_torch = cs::memory::to_torch(scheduler, target);
  target.wait();
  x_torch.set_requires_grad(true);
  const auto loss_torch = at::cross_entropy_loss(x_torch, target_torch);
  loss_torch.backward();
  ASSERT_TRUE(at::allclose(loss_torch, loss_ref_torch));
  ASSERT_TRUE(at::allclose(x_torch.grad(), dx));
}

TEST_F(TestCrossEntropyFixture, TestF32) { Test<float>(128); }
TEST_F(TestCrossEntropyFixture, TestF64) { Test<double>(128); }

template <typename T>
void TestCrossEntropyFixture::TestAmp(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  auto x =
      cs::compute::Utils::rand(scheduler, {size, 2 * size, 3 * size}, option);
  x = cs::compute::Utils::view(scheduler, x, {-1, x.size(-1)});
  auto target = cs::compute::Utils::randint(
      scheduler, 0, 3 * size, {size, 2 * size}, option.dtype(at::kLong));
  target = cs::compute::Utils::view(scheduler, target, {-1});
  auto state = cs::compute::CrossEntropy::init(scheduler);

  at::Tensor loss_ref_torch, loss_torch, x_torch;
  {
    cs::autocast::ContextGuard guard{scheduler, dtype};
    auto loss = cs::compute::CrossEntropy::forward(scheduler, state, x, target);
    loss_ref_torch = cs::memory::to_torch(scheduler, loss);
    loss.wait();
    x_torch = cs::memory::to_torch(scheduler, x);
    x.wait();
    auto target_torch = cs::memory::to_torch(scheduler, target);
    target.wait();
    x_torch = x_torch.to(at::kFloat);
    x_torch.set_requires_grad(true);
    loss_torch = at::cross_entropy_loss(x_torch, target_torch);
  }

  auto dx = cs::compute::CrossEntropy::backward(scheduler, state);
  auto dx_torch = cs::memory::to_torch(scheduler, dx);
  dx.wait();
  loss_torch.backward();
  ASSERT_TRUE(at::allclose(loss_torch, loss_ref_torch));
  ASSERT_TRUE(at::allclose(x_torch.grad().to(at::kFloat), dx));
}

TEST_F(TestCrossEntropyFixture, TestAmpF16) { TestAmp<half>(128); }
TEST_F(TestCrossEntropyFixture, TestAmpBF16) { TestAmp<__nv_bfloat16>(128); }
