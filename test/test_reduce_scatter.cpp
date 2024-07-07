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
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/reduce_scatter.h"
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

class ReduceScatterNCCLTestFixture : public ::testing::Test {
 protected:
  cs::communication::Comm comm{
      cs::communication::get_comm_world(cs::communication::kNCCL)};
  cs::DynamicScheduler scheduler{static_cast<int>(comm.get_rank())};

  ReduceScatterNCCLTestFixture() {
    CS_CHECK_CUDART(cudaSetDevice(comm.get_rank()));
  }

  template <typename T>
  void TestlAllToAllT(int blockSize);
};

template <typename T>
void ReduceScatterNCCLTestFixture::TestlAllToAllT(const int blockSize) {
  const at::Device device(at::kCUDA, comm.get_rank());
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  at::manual_seed(comm.get_rank() + 1);
  const int m = blockSize * comm.get_size();
  std::vector<cs::ReadOnlyTensor> vs;
  vs.reserve(comm.get_size());
  for (int i = 0; i < comm.get_size(); ++i) {
    auto t = cs::compute::Utils::rand(scheduler, {blockSize}, option);
    vs.push_back(t);
  }
  auto r = cs::compute::Utils::empty(scheduler, {blockSize}, option);
  cs::communication::ReduceScatter::run(scheduler, comm, {r}, {vs},
                                        cs::communication::kSUM);

  auto x_torch = cs::memory::to_torch(scheduler, r);
  r.wait();

  auto accumulator = torch::zeros({m}, option);
  for (int i = 0; i < comm.get_size(); ++i) {
    at::manual_seed(i + 1);
    for (int j = 0; j < comm.get_size(); ++j) {
      const auto full_random = torch::rand({blockSize}, option);
      accumulator.narrow(0, j * blockSize, blockSize) += full_random;
    }
  }
  ASSERT_TRUE(at::allclose(
      accumulator.narrow(0, comm.get_rank() * blockSize, blockSize), x_torch));
}

TEST_F(ReduceScatterNCCLTestFixture, TestForwardF32) {
  TestlAllToAllT<float>(128);
}
TEST_F(ReduceScatterNCCLTestFixture, TestForwardF64) {
  TestlAllToAllT<double>(128);
}
