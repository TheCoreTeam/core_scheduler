#include <ATen/ATen.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

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
template <typename T>
void TestForwardT(const dllm::ContextCompute &context) {
  auto x = at::rand({5, 5});
  auto y = at::rand({5, 5});
  auto z = x + y;
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
