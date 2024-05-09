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
  auto B = 4, T = 128, n_head = 32, n_embd = 256;
  // Initialize random tensors for q, k, v
  auto qkv = torch::randn({B, T, n_embd * 3}, torch::dtype(torch::kFloat16))
                 .split(n_embd, -1);
  auto q = qkv[0].view({B, T, n_head, n_embd / n_head});
  auto k = qkv[1].view({B, T, n_head, n_embd / n_head});
  auto v = qkv[2].view({B, T, n_head, n_embd / n_head});

  // Transpose for attention calculation
  q = q.transpose(1, 2);
  k = k.transpose(1, 2);
  v = v.transpose(1, 2);

  // Create lower triangular mask
  auto mask = torch::tril(torch::ones({T, T}, torch::dtype(torch::kFloat16)))
                  .view({1, 1, T, T});

  // Calculate scaled dot product attention
  auto att =
      torch::matmul(q, k.transpose(-2, -1)) * (1.0 / std::sqrt(k.size(-1)));
  att = att.masked_fill(mask == 0, -std::numeric_limits<float>::infinity());
  att = torch::softmax(att, -1);
  att = torch::dropout(att, /*p=*/0.0, /*train=*/false);

  // Apply attention to values
  auto y_attn = torch::matmul(att, v);
  y_attn = y_attn.transpose(1, 2);
}
}  // namespace

TEST_F(FlashAttentionTestFixture, TestForwardF16F32F32) {
  TestForwardT<nv_half>(context);
}
TEST_F(FlashAttentionTestFixture, TestForwardF32F32F32) {
  TestForwardT<float>(context);
}
TEST_F(FlashAttentionTestFixture, TestForwardF64F64F64) {
  TestForwardT<double>(context);
}
