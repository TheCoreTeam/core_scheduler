#include <gtest/gtest.h>
#include <torch/torch.h>

#include "compute/utils.h"
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

class TestDLLMInit : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute tp{0, 2};
  dllm::ThreadStreamCudart stream{0};

  template <typename Element>
  void TestRoutine(int T);
};

template <typename Element>
void TestDLLMInit::TestRoutine(const int T) {
  const int B = 2;
  const torch::Device device = torch::kCUDA;
  const torch::Dtype dtype = TypeToTorch<Element>::type;
  const auto option = torch::TensorOptions().dtype(dtype).device(device);

  torch::Tensor xRef;
  const auto x = dllm::Tensor::create();
  torch::manual_seed(1);
  {
    auto task = dllm::compute::Utils::randn(x, {B, T}, option);
    tp.submit(std::move(task));
  }

  {
    auto task = dllm::memory::toTorch(xRef, x);
    stream.submit(std::move(task));
    x->wait();
  }
  torch::manual_seed(1);

  ASSERT_TRUE(at::allclose(xRef, torch::randn({B, T}, option)));
}

TEST_F(TestDLLMInit, TestRand_512) { TestRoutine<float>(512); }
