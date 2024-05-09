#include <ATen/ATen.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/nn/functional.h>

#include <Eigen/Dense>

#include "logger.h"
#include "threading/thread_pool_compute.h"
#include "util.h"

namespace Eigen::internal {
template <>
struct scalar_random_op<nv_half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_random_op)
  inline const nv_half operator()() const {
    return static_cast<nv_half>(random<float>());
  }
};
}  // namespace Eigen::internal

class FlashAttentionTestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};

  void SetUp() override {
    CHECK_CUDART(
        cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
    CHECK_CUBLAS(cublasCreate_v2(&context.cublasHandle));
    CHECK_CUBLAS(cublasSetStream_v2(context.cublasHandle, context.cudaStream));
  }

  void TearDown() override {
    CHECK_CUBLAS(cublasDestroy_v2(context.cublasHandle));
    CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
  }
};

namespace {
template <typename ComputeType>
cublasComputeType_t toCublasComputeType() {
  if constexpr (std::is_same_v<ComputeType, double>) {
    return CUBLAS_COMPUTE_64F_PEDANTIC;
  } else if constexpr (std::is_same_v<ComputeType, float>) {
    return CUBLAS_COMPUTE_32F_PEDANTIC;
  }
}
}  // namespace

namespace {
template <typename Dtype>
void TestForwardT(const dllm::ContextCompute &context) {
  auto B = 4, T = 1024, n_head = 32, n_embd = 1024;
  auto q = at::randn({B, T, n_embd}, at::TensorOptions{}
                                         .dtype(c10::ScalarType::Half)
                                         .device(c10::DeviceType::CUDA));
  auto k = at::randn({B, T, n_embd}, at::TensorOptions{}
                                         .dtype(c10::ScalarType::Half)
                                         .device(c10::DeviceType::CUDA));
  auto v = at::randn({B, T, n_embd}, at::TensorOptions{}
                                         .dtype(c10::ScalarType::Half)
                                         .device(c10::DeviceType::CUDA));
  k = k.view({B, T, n_head, n_embd / n_head});  // (B, T, nh, hs)
  q = q.view({B, T, n_head, n_embd / n_head});  // (B, T, nh, hs)
  v = v.view({B, T, n_head, n_embd / n_head});  // (B, T, nh, hs)
  auto scale = 1.0 / std::sqrt(static_cast<double>(k.size(-1)));
  q = q.transpose(1, 2);
  k = k.transpose(1, 2);
  v = v.transpose(1, 2);
  auto mask =
      at::tril(at::ones({1, 1, T, T}, at::TensorOptions{}
                                          .dtype(c10::ScalarType::Half)
                                          .device(c10::DeviceType::CUDA)));
  auto att = at::matmul(q, k.transpose(-1, -1)) * scale;
  att = att.masked_fill(
      mask.index({"...", "...",
                  torch::indexing::Slice(torch::indexing::None, T),
                  torch::indexing::Slice(torch::indexing::None, T)}) == 0,
      -std::numeric_limits<float>::infinity());
  att = torch::nn::functional::softmax(att, -1);
  att = torch::nn::functional::dropout(
      att, torch::nn::functional::DropoutFuncOptions{}.p(0.0));
  auto y_attn = at::matmul(att, v);
  y_attn = y_attn.transpose(1, 2);
}
}  // namespace

TEST_F(FlashAttentionTestFixture, TestForwardF16F32F32) {
  // TestForwardT<nv_half>(context);
}
TEST_F(FlashAttentionTestFixture, TestForwardF32F32F32) {
  // TestForwardT<float>(context);
}
TEST_F(FlashAttentionTestFixture, TestForwardF64F64F64) {
  // TestForwardT<double>(context);
}
