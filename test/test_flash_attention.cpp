#include <ATen/ATen.h>
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

namespace {
template <typename Dtype>
void TestT(const dllm::ContextCompute &context) {
  static_assert(std::is_same_v<Dtype, nv_half> ||
                std::is_same_v<Dtype, nv_bfloat16>);
  auto dtype =
      std::is_same_v<Dtype, nv_half> ? torch::kFloat16 : torch::kBFloat16;
  torch::manual_seed(1);
  dllm::TensorIndexType B = 1, T = 2, n_head = 2, n_embd = n_head * 8;
  // Initialize random tensors for q, k, v
  auto qkv = torch::randn({B, T, n_embd * 3}, torch::dtype(torch::kFloat16))
                 .split(n_embd, -1);
  auto q = qkv[0]
               .view({B, T, n_head, n_embd / n_head})
               .contiguous()
               .requires_grad_(true);
  auto k = qkv[1]
               .view({B, T, n_head, n_embd / n_head})
               .contiguous()
               .requires_grad_(true);
  auto v = qkv[2]
               .view({B, T, n_head, n_embd / n_head})
               .contiguous()
               .requires_grad_(true);

  void *randomDevicePtr;
  auto randomShape = cute::make_shape(static_cast<dllm::TensorIndexType>(2));
  auto randomLayout = cute::make_layout(randomShape, cute::GenRowMajor{});
  CHECK_CUDART(
      cudaMalloc(&randomDevicePtr, sizeof(int64_t) * cute::size(randomLayout)));
  auto randomTensor = std::make_shared<dllm::Tensor1D>(
      randomDevicePtr, randomLayout, dllm::toDtype<int64_t>(), dllm::CUDA);

  void *qDevicePtr;
  auto qShape = cute::make_shape(B, T, n_head, n_embd / n_head);
  auto qLayout = cute::make_layout(qShape, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&qDevicePtr, sizeof(Dtype) * cute::size(qLayout)));
  CHECK_CUDART(cudaMemcpy(qDevicePtr, q.data_ptr(),
                          sizeof(Dtype) * cute::size(qLayout),
                          cudaMemcpyHostToDevice));
  auto qTensor = std::make_shared<dllm::Tensor4D>(
      qDevicePtr, qLayout, dllm::toDtype<Dtype>(), dllm::CUDA);

  void *kDevicePtr;
  auto kShape = cute::make_shape(B, T, n_head, n_embd / n_head);
  auto kLayout = cute::make_layout(kShape, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&kDevicePtr, sizeof(Dtype) * cute::size(kLayout)));
  CHECK_CUDART(cudaMemcpy(kDevicePtr, k.data_ptr(),
                          sizeof(Dtype) * cute::size(kLayout),
                          cudaMemcpyHostToDevice));
  auto kTensor = std::make_shared<dllm::Tensor4D>(
      kDevicePtr, kLayout, dllm::toDtype<Dtype>(), dllm::CUDA);

  void *vDevicePtr;
  auto vShape = cute::make_shape(B, T, n_head, n_embd / n_head);
  auto vLayout = cute::make_layout(vShape, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&vDevicePtr, sizeof(Dtype) * cute::size(vLayout)));
  CHECK_CUDART(cudaMemcpy(vDevicePtr, v.data_ptr(),
                          sizeof(Dtype) * cute::size(vLayout),
                          cudaMemcpyHostToDevice));
  auto vTensor = std::make_shared<dllm::Tensor4D>(
      vDevicePtr, vLayout, dllm::toDtype<Dtype>(), dllm::CUDA);

  void *outDevicePtr;
  auto outShape = cute::make_shape(B, T, n_head, n_embd / n_head);
  auto outLayout = cute::make_layout(outShape, cute::GenRowMajor{});
  CHECK_CUDART(
      cudaMalloc(&outDevicePtr, sizeof(Dtype) * cute::size(outLayout)));
  auto outTensor = std::make_shared<dllm::Tensor4D>(
      outDevicePtr, outLayout, dllm::toDtype<Dtype>(), dllm::CUDA);

  void *softmaxDevicePtr;
  auto softmaxShape = cute::make_shape(B, n_head, T);
  auto softmaxLayout = cute::make_layout(softmaxShape, cute::GenRowMajor{});
  CHECK_CUDART(
      cudaMalloc(&softmaxDevicePtr, sizeof(float) * cute::size(softmaxLayout)));
  auto softmaxTensor = std::make_shared<dllm::Tensor3D>(
      softmaxDevicePtr, softmaxLayout, dllm::toDtype<float>(), dllm::CUDA);
  CHECK_CUDART(cudaDeviceSynchronize());

  // Transpose for attention calculation
  auto qT = q.transpose(1, 2);
  auto kT = k.transpose(1, 2);
  auto vT = v.transpose(1, 2);

  // Create lower triangular mask
  auto mask = torch::tril(torch::ones({T, T}, torch::dtype(torch::kFloat16)))
                  .view({1, 1, T, T});

  // Calculate scaled dot product attention
  auto scale = 1.0 / std::sqrt(kT.size(-1));
  auto att = torch::matmul(qT, kT.transpose(-2, -1)) * scale;
  att = att.masked_fill(mask == 0, -std::numeric_limits<float>::infinity());
  att = torch::softmax(att, -1);
  att = torch::dropout(att, /*p=*/0.0, /*train=*/false);

  // Apply attention to values
  auto y_attn = torch::matmul(att, vT);
  y_attn = y_attn.transpose(1, 2).contiguous();

  {
    auto task = dllm::compute::FlashAttention::forward(
        randomTensor, outTensor, softmaxTensor, qTensor, kTensor, vTensor, 0.0,
        scale);
    task(&context);
    outTensor->future->wait();
  }

  auto outRef = at::empty_like(y_attn);
  ASSERT_TRUE(outRef.sizes() == q.sizes());
  CHECK_CUDART(cudaMemcpy(outRef.data_ptr(), outTensor->data(),
                          sizeof(Dtype) * cute::size(outTensor->layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());
  std::cout << y_attn << std::endl;
  std::cout << outRef << std::endl;
  ASSERT_TRUE(torch::allclose(y_attn, outRef, 1e-5, 1e-2));

  void *dqDevicePtr;
  auto dqShape = cute::make_shape(B, T, n_head, n_embd / n_head);
  auto dqLayout = cute::make_layout(dqShape, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&dqDevicePtr, sizeof(Dtype) * cute::size(dqLayout)));
  auto dqTensor = std::make_shared<dllm::Tensor4D>(
      dqDevicePtr, dqLayout, dllm::toDtype<Dtype>(), dllm::CUDA);

  void *dkDevicePtr;
  auto dkShape = cute::make_shape(B, T, n_head, n_embd / n_head);
  auto dkLayout = cute::make_layout(dkShape, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&dkDevicePtr, sizeof(Dtype) * cute::size(dkLayout)));
  auto dkTensor = std::make_shared<dllm::Tensor4D>(
      dkDevicePtr, dkLayout, dllm::toDtype<Dtype>(), dllm::CUDA);

  void *dvDevicePtr;
  auto dvShape = cute::make_shape(B, T, n_head, n_embd / n_head);
  auto dvLayout = cute::make_layout(dvShape, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&dvDevicePtr, sizeof(Dtype) * cute::size(dvLayout)));
  auto dvTensor = std::make_shared<dllm::Tensor4D>(
      dvDevicePtr, dvLayout, dllm::toDtype<Dtype>(), dllm::CUDA);

  auto dout = torch::ones_like(outRef);
  void *doutDevicePtr;
  auto doutShape = cute::make_shape(B, T, n_head, n_embd / n_head);
  auto doutLayout = cute::make_layout(doutShape, cute::GenRowMajor{});
  CHECK_CUDART(
      cudaMalloc(&doutDevicePtr, sizeof(Dtype) * cute::size(doutLayout)));
  auto doutTensor = std::make_shared<dllm::Tensor4D>(
      doutDevicePtr, doutLayout, dllm::toDtype<Dtype>(), dllm::CUDA);
  CHECK_CUDART(cudaMemcpy(doutTensor->data(), dout.data_ptr(),
                          sizeof(Dtype) * cute::size(doutTensor->layout),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  {
    auto task = dllm::compute::FlashAttention::backward(
        dqTensor, dkTensor, dvTensor, doutTensor, randomTensor, outTensor,
        softmaxTensor, qTensor, kTensor, vTensor, 0.0, scale);
    task(&context);
    dqTensor->future->wait();
  }

  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);

  CHECK_CUDART(cudaMemcpy(dq.data_ptr(), dqTensor->data(),
                          sizeof(Dtype) * cute::size(dqTensor->layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(dk.data_ptr(), dkTensor->data(),
                          sizeof(Dtype) * cute::size(dkTensor->layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(dv.data_ptr(), dvTensor->data(),
                          sizeof(Dtype) * cute::size(dvTensor->layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  y_attn.backward(dout);
    std::cout << torch::allclose(dq, q.grad(), 1e-5, 1e-2) << std::endl;
    std::cout << torch::allclose(dk, k.grad(), 1e-5, 1e-2) << std::endl;
    std::cout << torch::allclose(dv, v.grad(), 1e-5, 1e-2) << std::endl;
    std::cout << dq << std::endl;
    std::cout << q.grad() << std::endl;
    std::cout << dk << std::endl;
    std::cout << k.grad() << std::endl;
    std::cout << dv << std::endl;
    std::cout << v.grad() << std::endl;
  ASSERT_TRUE(torch::allclose(dq, q.grad(), 1e-5, 1e-2));
  ASSERT_TRUE(torch::allclose(dk, k.grad(), 1e-5, 1e-2));
  ASSERT_TRUE(torch::allclose(dv, v.grad(), 1e-5, 1e-2));

  CHECK_CUDART(cudaFree(dqDevicePtr));
  CHECK_CUDART(cudaFree(dkDevicePtr));
  CHECK_CUDART(cudaFree(dvDevicePtr));
  CHECK_CUDART(cudaFree(doutDevicePtr));

  CHECK_CUDART(cudaFree(randomDevicePtr));
  CHECK_CUDART(cudaFree(qDevicePtr));
  CHECK_CUDART(cudaFree(kDevicePtr));
  CHECK_CUDART(cudaFree(vDevicePtr));
  CHECK_CUDART(cudaFree(outDevicePtr));
  CHECK_CUDART(cudaFree(softmaxDevicePtr));
}
}  // namespace

TEST_F(FlashAttentionTestFixture, TestF16) { TestT<nv_half>(context); }
TEST_F(FlashAttentionTestFixture, TestBF16) { TestT<nv_bfloat16>(context); }
