#include <ATen/ATen.h>
#include <compute/fused_classifier.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/nn/functional.h>

#include <Eigen/Dense>

#include "compute/flash_attation.h"
#include "logger.h"
#include "threading/thread_pool_compute.h"
#include "util.h"

namespace Eigen::internal {
template <>
struct scalar_random_op<nv_half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_random_op)
  nv_half operator()() const { return static_cast<nv_half>(random<float>()); }
};
}  // namespace Eigen::internal

class FusedClassifierTestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};

  void SetUp() override {
    CHECK_CUDART(
        cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
    CHECK_CUBLAS(cublasCreate_v2(&context.cublasHandle));
    CHECK_CUBLAS(cublasSetStream_v2(context.cublasHandle, context.cudaStream));
    CHECK_CUDART(cudaDeviceGetDefaultMemPool(&context.memPool, 0));
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

template <typename T>
struct ToTorchType;

template <>
struct ToTorchType<nv_half> {
  constexpr static auto value = torch::kHalf;
};

template <>
struct ToTorchType<float> {
  constexpr static auto value = torch::kFloat;
};

template <>
struct ToTorchType<nv_bfloat16> {
  constexpr static auto value = torch::kBFloat16;
};

namespace {
template <typename Dtype>
void TestT(const dllm::ContextCompute &context) {
  torch::manual_seed(1);
  const dllm::TensorIndexType B = 15, T = 7, V = 251, VP = 256;
  const auto dtype = torch::TensorOptions{ToTorchType<Dtype>::value};
  auto logits = torch::randn({B, T, VP}, torch::kDouble);
  auto targets = torch::randint(0, V, {B, T}, torch::kInt32);

  logits.slice(2, V, VP).fill_(0.);
  logits = logits.requires_grad_(true);
  auto valid_logits = logits.slice(2, 0, V);

  auto max_logits = std::get<0>(valid_logits.max(-1, true));
  auto logits_stable = valid_logits - max_logits;

  auto exp_logits = logits_stable.exp();
  auto sum_exp_logits = exp_logits.sum(-1, true);
  auto prob = exp_logits / sum_exp_logits;

  auto prob_target =
      prob.gather(2, targets.to(torch::kInt64).unsqueeze(2)).squeeze(2);

  auto losses = -prob_target.log();
  losses.backward(torch::ones_like(losses) / (B * T));

  void *logitsDevicePtr;
  auto logitsShape = cute::make_shape(B, T, V);
  auto logitsStride = cute::make_shape(T * VP, VP, cute::_1{});
  auto logitsLayout = cute::make_layout(logitsShape, logitsStride);
  CHECK_CUDART(
      cudaMalloc(&logitsDevicePtr, sizeof(Dtype) * cute::cosize(logitsLayout)));
  auto logitsCpy = logits.to(dtype);
  CHECK_CUDART(cudaMemcpy(logitsDevicePtr, logitsCpy.data_ptr(),
                          sizeof(Dtype) * cute::cosize(logitsLayout),
                          cudaMemcpyHostToDevice));
  auto logitsTensor = std::make_shared<dllm::Tensor3D>(
      logitsDevicePtr, logitsLayout, dllm::toDtype<Dtype>(), dllm::CUDA);

  void *lossesDevicePtr;
  auto lossesShape = cute::make_shape(B, T);
  auto lossesLayout = cute::make_layout(lossesShape, cute::GenRowMajor{});
  CHECK_CUDART(
      cudaMalloc(&lossesDevicePtr, sizeof(Dtype) * cute::size(lossesLayout)));
  auto lossesTensor = std::make_shared<dllm::Tensor2D>(
      lossesDevicePtr, lossesLayout, dllm::toDtype<Dtype>(), dllm::CUDA);

  void *targetsDevicePtr;
  auto targetsShape = cute::make_shape(B, T);
  auto targetsLayout = cute::make_layout(targetsShape, cute::GenRowMajor{});
  CHECK_CUDART(
      cudaMalloc(&targetsDevicePtr, sizeof(int) * cute::size(targetsLayout)));
  CHECK_CUDART(cudaMemcpy(targetsDevicePtr, targets.data_ptr<int>(),
                          sizeof(int) * cute::size(targetsLayout),
                          cudaMemcpyHostToDevice));
  auto targetsTensor = std::make_shared<dllm::Tensor2D>(
      targetsDevicePtr, targetsLayout, dllm::toDtype<int>(), dllm::CUDA);
  CHECK_CUDART(cudaDeviceSynchronize());

  {
    auto task = dllm::compute::FusedClassifier::call(logitsTensor, lossesTensor,
                                                     targetsTensor);
    task(&context);
    dllm::util::FutureGuard{logitsTensor->future->rFuture};
    dllm::util::FutureGuard{logitsTensor->future->wFuture};
  }

  auto lossesRef = at::empty_like(losses, dtype);
  CHECK_CUDART(cudaMemcpy(lossesRef.data_ptr(), lossesTensor->data(),
                          sizeof(Dtype) * cute::size(lossesTensor->layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto gradRef = at::empty_like(logits, dtype);
  CHECK_CUDART(cudaMemcpy(gradRef.data_ptr(), logitsTensor->data(),
                          sizeof(Dtype) * cute::cosize(logitsTensor->layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());
  double atol = 1e-5;
  if constexpr (std::is_same_v<Dtype, nv_half>) {
    atol = 2e-2;
  } else if constexpr (std::is_same_v<Dtype, nv_bfloat16>) {
    atol = 4e-2;
  }
  ASSERT_TRUE(
      torch::allclose(lossesRef.to(torch::kDouble), losses, 1e-5, atol));
  ASSERT_TRUE(
      torch::allclose(gradRef.to(torch::kDouble), logits.grad(), 1e-5, atol));

  CHECK_CUDART(cudaFree(logitsDevicePtr));
  CHECK_CUDART(cudaFree(lossesDevicePtr));
  CHECK_CUDART(cudaFree(targetsDevicePtr));
}
}  // namespace

TEST_F(FusedClassifierTestFixture, TestF32) { TestT<float>(context); }
TEST_F(FusedClassifierTestFixture, TestF16) { TestT<nv_half>(context); }
TEST_F(FusedClassifierTestFixture, TestBF16) { TestT<nv_bfloat16>(context); }
