#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <fstream>

#include "logger.h"
#include "memory/allocate.h"
#include "memory/cuda_memcpy.h"
#include "optimizer/adamw.h"
#include "tensor.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_cudart.h"
#include "util.h"

namespace Eigen::internal {
template <>
struct scalar_random_op<nv_half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_random_op)
  inline nv_half operator()() const {
    return static_cast<nv_half>(random<float>());
  }
};
}  // namespace Eigen::internal

class TestDLLMAdamW : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};

  void SetUp() override {
    CHECK_CUDART(
        cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
    CHECK_CUDART(cudaDeviceGetDefaultMemPool(&context.memPool, 0));
  }

  void TearDown() override {
    CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
  }

  template <typename Element>
  void TestRoutine(const dllm::TensorIndexType size);
};

template <typename Element>
void TestDLLMAdamW::TestRoutine(const dllm::TensorIndexType size) {
  const double lr = 1e-3;
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double eps = 1e-8;
  const double weight_decay = 1e-2;
  const long t = 0;
  Eigen::Array<Element, Eigen::Dynamic, 1> xHost(size), xHostRef(size);
  Eigen::Array<Element, Eigen::Dynamic, 1> dxHost(size);
  Eigen::Array<Element, Eigen::Dynamic, 1> mHost(size), mHostRef(size);
  Eigen::Array<Element, Eigen::Dynamic, 1> vHost(size), vHostRef(size);
  xHost.setRandom();
  dxHost.setRandom();
  mHost.setRandom();
  vHost.setRandom();
  vHost = vHost.abs();

  auto shape = cute::make_shape(size);
  auto layout = cute::make_layout(shape, cute::GenRowMajor{});

  CHECK_CUDART(cudaDeviceSynchronize());

  dllm::ThreadStreamCudart h2d{0}, d2h{0};
  dllm::ThreadPoolCompute tp{0, 1};

  std::shared_ptr<dllm::Tensor1D> xTensor;
  {
    auto task = dllm::memory::allocateRowMajor(
        xTensor, {size}, dllm::toDtype<Element>(), dllm::CUDA);
    h2d.submit(std::move(task));
  }
  std::shared_ptr<dllm::Tensor1D> dxTensor;
  {
    auto task = dllm::memory::allocateRowMajor(
        dxTensor, {size}, dllm::toDtype<Element>(), dllm::CUDA);
    h2d.submit(std::move(task));
  }

  using AdamW = dllm::optimizer::AdamW<>;
  using State = AdamW::State;
  std::shared_ptr<State> state;
  {
    auto task =
        dllm::optimizer::AdamW<>::init<dllm::R_32F, dllm::CUDA>(state, layout);
    tp.submit(std::move(task));
  }

  {
    auto task = dllm::memory::memcpyFromHost(state->m, mHost.data());
    h2d.submit(std::move(task));
  }
  {
    auto task = dllm::memory::memcpyFromHost(state->v, vHost.data());
    h2d.submit(std::move(task));
  }
  {
    auto task = dllm::memory::memcpyFromHost(xTensor, xHost.data());
    h2d.submit(std::move(task));
  }
  {
    auto task = dllm::memory::memcpyFromHost(dxTensor, dxHost.data());
    h2d.submit(std::move(task));
  }
  {
    auto task = dllm::optimizer::AdamW<false>::step(state, xTensor, dxTensor);
    tp.submit(std::move(task));
  }
  {
    auto task = dllm::memory::memcpyToHost(xHostRef.data(), xTensor);
    d2h.submit(std::move(task));
  }
  {
    auto task = dllm::memory::memcpyToHost(mHostRef.data(), state->m);
    d2h.submit(std::move(task));
  }
  {
    auto task = dllm::memory::memcpyToHost(vHostRef.data(), state->v);
    d2h.submit(std::move(task));
  }

  {
    dllm::util::FutureGuard{xTensor->future->rFuture};
    dllm::util::FutureGuard{xTensor->future->wFuture};
    dllm::util::FutureGuard{state->m->future->rFuture};
    dllm::util::FutureGuard{state->m->future->wFuture};
    dllm::util::FutureGuard{state->v->future->rFuture};
    dllm::util::FutureGuard{state->v->future->wFuture};
  }

  xHost = xHost - lr * weight_decay * xHost;
  mHost = beta1 * mHost + (1 - beta1) * dxHost;
  vHost = beta2 * vHost + (1 - beta2) * dxHost.square();
  auto m_hat = mHost / (1 - std::pow(beta1, t + 1));
  auto v_hat = vHost / (1 - std::pow(beta2, t + 1));
  xHost = xHost - lr * m_hat / (v_hat.sqrt() + eps);

  ASSERT_NEAR((xHost - xHostRef).matrix().norm(), 0, 1e-4);
  ASSERT_NEAR((mHost - mHostRef).matrix().norm(), 0, 1e-4);
  ASSERT_NEAR((vHost - vHostRef).matrix().norm(), 0, 1e-4);
}

TEST_F(TestDLLMAdamW, TestF32_128) { TestRoutine<float>(128); }

TEST_F(TestDLLMAdamW, TestF32_512) { TestRoutine<float>(512); }

TEST_F(TestDLLMAdamW, TestF32_1024) { TestRoutine<float>(1024); }
