#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <fstream>
#include <torch/torch.h>
#include <c10/util/Half.h>
#include <iostream>
#include <flash_attention/layer_norm/layer_norm.h>
#include "tensor.h"

template<typename T>
struct TypeToTorch;

template<>
struct TypeToTorch<float> {
  using Type = float;
  static const at::ScalarType dtype = torch::kFloat;
};

template<>
struct TypeToTorch<nv_half> {
  using Type = c10::Half;
  static const at::ScalarType dtype = torch::kHalf;
};

class TestFlashAttnLayerNorm : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};

  void SetUp() override {
    CHECK_CUDART(
        cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  }

  void TearDown() override {
    CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
  }

  template <typename Element>
  void TestRoutine();
};

template <typename Element>
void TestFlashAttnLayerNorm::TestRoutine() {

  torch::manual_seed(2024);

  const dllm::TensorIndexType B = 1;
  const dllm::TensorIndexType T = 1024;
  const dllm::TensorIndexType d = 512;

  const float eps = 1e-5;
  const float tol = 1e-2;
  torch::Device device = torch::kCPU;
  torch::Dtype dtype = TypeToTorch<Element>::dtype;
  auto weight = torch::randn(d, torch::TensorOptions().dtype(dtype).device(device));
  auto bias = torch::randn(d, torch::TensorOptions().dtype(dtype).device(device));
  auto input = torch::randn({T, d}, torch::TensorOptions().dtype(dtype).device(device));

  auto input1 = input.detach().clone().set_requires_grad(true);
  auto weight1 = weight.detach().clone().set_requires_grad(true);
  auto bias1 = bias.detach().clone().set_requires_grad(true);

  auto output1 = at::layer_norm(input1, {d}, weight1, bias1);

  void *DeviceZ, *DeviceMu, *DeviceRSigma, *DeviceX0, *DeviceGamma, *DeviceBeta;
  auto shapeInput = cute::make_shape(T, d);
  auto shapeWeight = cute::make_shape(d);
  auto shapeBias = cute::make_shape(d);
  auto shapeMu = cute::make_shape(T);
  auto shapeRSigma = cute::make_shape(T);
  auto layoutZ = cute::make_layout(shapeInput, cute::GenRowMajor{});
  auto layoutMu = cute::make_layout(shapeMu, cute::GenRowMajor{});
  auto layoutRSigma = cute::make_layout(shapeRSigma, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&DeviceZ, sizeof(Element) * cute::size(layoutZ)));
  CHECK_CUDART(cudaMalloc(&DeviceMu, sizeof(Element) * cute::size(layoutMu)));
  CHECK_CUDART(cudaMalloc(&DeviceRSigma, sizeof(Element) * cute::size(layoutRSigma)));

  CHECK_CUDART(
      cudaMemset(DeviceZ, 0, sizeof(Element) * cute::size(layoutZ)));
  CHECK_CUDART(
      cudaMemset(DeviceMu, 0, sizeof(Element) * cute::size(layoutMu)));
  CHECK_CUDART(
      cudaMemset(DeviceRSigma, 0, sizeof(Element) * cute::size(layoutRSigma)));

  auto tensorZ = std::make_shared<dllm::Tensor2D>(
      DeviceZ, layoutZ, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorMu = std::make_shared<dllm::Tensor1D>(
      DeviceMu, layoutMu, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorRSigma = std::make_shared<dllm::Tensor1D>(
      DeviceRSigma, layoutRSigma, dllm::toDtype<Element>(), dllm::CUDA);

  auto layoutX0 = cute::make_layout(shapeInput, cute::GenRowMajor{});
  auto layoutGamma = cute::make_layout(shapeWeight, cute::GenRowMajor{});
  auto layoutBeta = cute::make_layout(shapeBias, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&DeviceX0, sizeof(Element) * cute::size(layoutX0)));
  CHECK_CUDART(cudaMalloc(&DeviceGamma, sizeof(Element) * cute::size(layoutGamma)));
  CHECK_CUDART(cudaMalloc(&DeviceBeta, sizeof(Element) * cute::size(layoutBeta)));

  CHECK_CUDART(cudaMemcpy(DeviceX0, input.data_ptr<typename TypeToTorch<Element>::Type>(),
                          sizeof(Element) * cute::size(layoutX0),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(DeviceGamma, weight.data_ptr<typename TypeToTorch<Element>::Type>(),
                          sizeof(Element) * cute::size(layoutGamma),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(DeviceBeta, bias.data_ptr<typename TypeToTorch<Element>::Type>(),
                          sizeof(Element) * cute::size(layoutBeta),
                          cudaMemcpyHostToDevice));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorX0 = std::make_shared<dllm::Tensor2D>(
      DeviceX0, layoutX0, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorGamma = std::make_shared<dllm::Tensor1D>(
      DeviceGamma, layoutGamma, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorBeta = std::make_shared<dllm::Tensor1D>(
      DeviceBeta, layoutBeta, dllm::toDtype<Element>(), dllm::CUDA);

  auto tast = dllm::flash_attn::layer_norm::forward(tensorZ, tensorMu, tensorRSigma, tensorX0, tensorGamma, tensorBeta, eps);
  tast(&context);

  auto output2 = torch::empty_like(output1);
  auto weight2 = torch::empty_like(weight1);
  auto bias2 = torch::empty_like(bias1);

  CHECK_CUDART(cudaMemcpy(output2.data_ptr<typename TypeToTorch<Element>::Type>(), DeviceZ,
                          sizeof(Element) * cute::size(layoutZ),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(weight2.data_ptr<typename TypeToTorch<Element>::Type>(), DeviceGamma,
                          sizeof(Element) * cute::size(layoutGamma),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(bias2.data_ptr<typename TypeToTorch<Element>::Type>(), DeviceBeta,
                          sizeof(Element) * cute::size(layoutBeta),
                          cudaMemcpyDeviceToHost));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto isApprox_input = output1.allclose(output2, tol);
  auto isApprox_weight = weight1.allclose(weight2, tol);
  auto isApprox_bias = bias1.allclose(bias2, tol);

  if (!isApprox_input) {
      std::ofstream file_output_ref("output1.txt");
      file_output_ref << output1 << std::endl;
      file_output_ref.close();
      std::ofstream file_output("output2.txt");
      file_output << output2 << std::endl;
      file_output.close();
  }

  if (!isApprox_weight) {
      std::ofstream file_weight_ref("weight1.txt");
      file_weight_ref << weight1 << std::endl;
      file_weight_ref.close();
      std::ofstream file_weight("weight2.txt");
      file_weight << weight2 << std::endl;
      file_weight.close();
  }

  if (!isApprox_bias) {
      std::ofstream file_bias_ref("bias1.txt");
      file_bias_ref << bias1 << std::endl;
      file_bias_ref.close();
      std::ofstream file_bias("bias2.txt");
      file_bias << bias2 << std::endl;
      file_bias.close();
  }

  ASSERT_TRUE(isApprox_input);
  ASSERT_TRUE(isApprox_weight);
  ASSERT_TRUE(isApprox_bias);
  CHECK_CUDART(cudaFree(DeviceZ));
  CHECK_CUDART(cudaFree(DeviceMu));
  CHECK_CUDART(cudaFree(DeviceRSigma));
  CHECK_CUDART(cudaFree(DeviceX0));
  CHECK_CUDART(cudaFree(DeviceGamma));
  CHECK_CUDART(cudaFree(DeviceBeta));
}

TEST_F(TestFlashAttnLayerNorm, TestFloat) {
  TestRoutine<float>();
}

//TEST_F(TestFlashAttnLayerNorm, TestHalf) {
//  TestRoutine<nv_half>();
//}
