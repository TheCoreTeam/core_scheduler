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
#include <ATen/ops/cat.h>
#include <ATen/ops/sum.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

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

class TestUtilsFixture : public ::testing::Test {
 protected:
  cs::DynamicScheduler scheduler{0};

  template <typename T>
  void TestCat(int size);

  template <typename T>
  void TestSum(int size);

  template <typename T>
  void TestClipNorm(int size);
};

template <typename T>
void TestUtilsFixture::TestCat(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  auto x1 = cs::compute::Utils::rand(scheduler, {size, size, size}, option);
  auto x2 = cs::compute::Utils::rand(scheduler, {size, size, size}, option);
  auto x3 = cs::compute::Utils::rand(scheduler, {size, size, size}, option);
  auto y = cs::compute::Utils::cat(scheduler, {x1, x2, x3}, -1);
  auto x1_torch = cs::memory::to_torch(scheduler, x1);
  x1.wait();
  auto x2_torch = cs::memory::to_torch(scheduler, x2);
  x2.wait();
  auto x3_torch = cs::memory::to_torch(scheduler, x3);
  x3.wait();
  auto y_torch = cs::memory::to_torch(scheduler, y);
  y.wait();
  ASSERT_TRUE(at::allclose(y, at::cat({x1_torch, x2_torch, x3_torch}, -1)));
}

TEST_F(TestUtilsFixture, TestCatF32) { TestCat<float>(128); }
TEST_F(TestUtilsFixture, TestCatF64) { TestCat<double>(128); }

template <typename T>
void TestUtilsFixture::TestSum(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  auto x = cs::compute::Utils::rand(scheduler, {size, size, size}, option);
  auto y = cs::compute::Utils::sum(scheduler, x, 0);
  auto x_torch = cs::memory::to_torch(scheduler, x);
  x.wait();
  auto y_torch = cs::memory::to_torch(scheduler, y);
  y.wait();
  ASSERT_TRUE(at::allclose(y, at::sum(x_torch, 0)));
}

TEST_F(TestUtilsFixture, TestSumF32) { TestSum<float>(128); }
TEST_F(TestUtilsFixture, TestSumF64) { TestSum<double>(128); }

template <typename T>
void TestUtilsFixture::TestClipNorm(const int size) {
  at::manual_seed(1);
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  auto x = cs::compute::Utils::rand(scheduler, {size, size, size}, option);
  auto y = cs::compute::Utils::rand(scheduler, {size, 2 * size, size}, option);
  auto x_torch = cs::memory::to_torch(scheduler, x);
  auto y_torch = cs::memory::to_torch(scheduler, y);
  const_cast<at::Tensor &>(x_torch.grad()) = x_torch;
  const_cast<at::Tensor &>(y_torch.grad()) = y_torch;
  torch::nn::utils::clip_grad_norm_(std::vector{x_torch, y_torch}, 1., 2.);
  auto total_norm =
      cs::compute::Utils::clip_grad_norm_(scheduler, {x, y}, 1, 2);
  ASSERT_TRUE(at::allclose(x_torch, x));
  ASSERT_TRUE(at::allclose(y_torch, y));
}

TEST_F(TestUtilsFixture, TestClipNormF32) { TestClipNorm<float>(128); }
TEST_F(TestUtilsFixture, TestClipNormF64) { TestClipNorm<double>(128); }
