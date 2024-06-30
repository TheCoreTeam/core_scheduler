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

#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/all_reduce.h"
#include "compute/utils.h"
#include "helper.h"
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

class AllReduceNcclTestFixture : public ::testing::Test {
 protected:
  cs::communication::Comm comm{
      cs::communication::getCommWorld(cs::communication::NCCL)};
  cs::DynamicScheduler scheduler{static_cast<int>(comm.getRank())};

  AllReduceNcclTestFixture() { CHECK_CUDART(cudaSetDevice(comm.getRank())); }

  template <typename T>
  void TestAllReduceT(int m);
};

template <typename T>
void AllReduceNcclTestFixture::TestAllReduceT(const int m) {
  const at::Device device(at::kCUDA, comm.getRank());
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  at::manual_seed(comm.getRank() + 1);
  auto x = cs::compute::Utils::rand(scheduler, {m}, option);
  auto y = cs::compute::Utils::rand(scheduler, {m}, option);
  cs::communication::AllReduce::runInplace(scheduler, comm, {x, y},
                                           cs::communication::SUM);

  auto x_torch = cs::memory::toTorch(scheduler, x);
  auto y_torch = cs::memory::toTorch(scheduler, y);
  x.wait();
  y.wait();

  auto accumulator_x = torch::zeros_like(x_torch);
  auto accumulator_y = torch::zeros_like(y_torch);
  for (int i = 0; i < comm.getSize(); ++i) {
    at::manual_seed(i + 1);
    accumulator_x += torch::rand({m}, option);
    accumulator_y += torch::rand({m}, option);
  }
  GTEST_ASSERT_TRUE(at::allclose(x, x_torch));
  GTEST_ASSERT_TRUE(at::allclose(y, y_torch));
}

TEST_F(AllReduceNcclTestFixture, TestForwardF32) { TestAllReduceT<float>(128); }
TEST_F(AllReduceNcclTestFixture, TestForwardF64) {
  TestAllReduceT<double>(128);
}
