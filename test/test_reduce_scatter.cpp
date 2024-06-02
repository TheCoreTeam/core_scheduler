#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/reduce_scatter.h"
#include "compute/utils.h"
#include "logger.h"
#include "memory/to_torch.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_cudart.h"
#include "threading/thread_stream_mpi.h"
#include "threading/thread_stream_nccl.h"

namespace dllm::test {
ThreadStreamNccl *getNcclStream();
}  // namespace dllm::test

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
  dllm::ContextCompute context{};
  dllm::ThreadStreamCudart *copy;
  dllm::ThreadPoolCompute *tp;

  ReduceScatterNCCLTestFixture() {
    copy = new dllm::ThreadStreamCudart{0};
    tp = new dllm::ThreadPoolCompute{0, 3};
    CHECK_CUDART(cudaSetDevice(0));
  }

  ~ReduceScatterNCCLTestFixture() {
    delete tp;
    delete copy;
  }

  template <typename T>
  void TestlAllToAllT(int blockSize);
};

template <typename T>
void ReduceScatterNCCLTestFixture::TestlAllToAllT(const int blockSize) {
  const auto stream = dllm::test::getNcclStream();
  const at::Device device(at::kCUDA, 0);
  const at::ScalarType dtype = TypeToTorch<T>::type;
  const auto option = at::TensorOptions().dtype(dtype).device(device);
  const int m = blockSize * stream->commSize();
  at::manual_seed(stream->rank() + 1);
  std::vector<std::shared_ptr<const dllm::ReadOnlyTensor>> vs;
  vs.reserve(stream->commSize());
  for (int i = 0; i < stream->commSize(); ++i) {
    auto t = dllm::Tensor::create();
    auto task = dllm::compute::Utils::rand(t, {blockSize}, option);
    tp->submit(std::move(task));
    vs.push_back(t);
  }
  auto r = dllm::Tensor::create();
  {
    auto task = dllm::compute::Utils::empty(r, {blockSize}, option);
    tp->submit(std::move(task));
  }
  {
    auto task =
        dllm::communication::ReduceScatter<dllm::communication::NCCL>::run(
            r, vs, dllm::communication::SUM);
    stream->submit(std::move(task));
  }

  at::Tensor x_torch;
  {
    auto task = dllm::memory::toTorch(x_torch, r);
    copy->submit(std::move(task));
    r->wait();
  }

  auto accumulator = torch::zeros({m}, option);
  for (int i = 0; i < stream->commSize(); ++i) {
    at::manual_seed(i + 1);
    for (int j = 0; j < stream->commSize(); ++j) {
      const auto full_random = torch::rand({blockSize}, option);
      accumulator.narrow(0, j * blockSize, blockSize) += full_random;
    }
  }
  ASSERT_TRUE(at::allclose(
      accumulator.narrow(0, stream->rank() * blockSize, blockSize), x_torch));
}

TEST_F(ReduceScatterNCCLTestFixture, TestForwardF32) {
  TestlAllToAllT<float>(128);
}
TEST_F(ReduceScatterNCCLTestFixture, TestForwardF64) {
  TestlAllToAllT<double>(128);
}
