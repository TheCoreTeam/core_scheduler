#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <flash_attention/layer_norm/layer_norm.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "tensor.h"

template <typename T>
struct TypeToTorch;

template <>
struct TypeToTorch<float> {
  using Type = float;
  static const at::ScalarType dtype = torch::kFloat;
};

template <>
struct TypeToTorch<nv_half> {
  using Type = c10::Half;
  static const at::ScalarType dtype = torch::kHalf;
};

class TestFlashAttnLayerNorm : public ::testing::Test {
 protected:
  using Element = float;

  dllm::ContextCompute context{};

  const dllm::TensorIndexType B = 1;
  const dllm::TensorIndexType T = 1024;
  const dllm::TensorIndexType d = 512;

  const float eps = 1e-5;
  const float tol = 1e-2;

  torch::Device device = torch::kCPU;
  torch::Dtype dtype = TypeToTorch<Element>::dtype;

  at::Tensor weight;
  at::Tensor bias;
  at::Tensor input;

  at::Tensor input1;
  at::Tensor weight1;
  at::Tensor bias1;
  at::Tensor output1;

  at::Tensor grad_output;

  at::Tensor grad_input1;
  at::Tensor grad_weight1;
  at::Tensor grad_bias1;

  void *DeviceMu, *DeviceRSigma;
  std::shared_ptr<dllm::Tensor1D> tensorMu, tensorRSigma;

  void SetUp() override {
    CHECK_CUDART(
        cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
    CHECK_CUDART(cudaDeviceGetDefaultMemPool(&context.memPool, 0));
    torch::manual_seed(2024);

    weight =
        torch::randn(d, torch::TensorOptions().dtype(dtype).device(device));
    bias = torch::randn(d, torch::TensorOptions().dtype(dtype).device(device));
    input = torch::randn({T, d},
                         torch::TensorOptions().dtype(dtype).device(device));

    input1 = input.detach().clone().set_requires_grad(true);
    weight1 = weight.detach().clone().set_requires_grad(true);
    bias1 = bias.detach().clone().set_requires_grad(true);

    output1 = at::layer_norm(input1, {d}, weight1, bias1);

    grad_output = torch::randn_like(output1);
    auto grads = torch::autograd::grad({output1}, {input1, weight1, bias1},
                                       {grad_output}, true);
    grad_input1 = grads[0];
    grad_weight1 = grads[1];
    grad_bias1 = grads[2];

    auto shapeMu = cute::make_shape(T);
    auto shapeRSigma = cute::make_shape(T);
    auto layoutMu = cute::make_layout(shapeMu, cute::GenRowMajor{});
    auto layoutRSigma = cute::make_layout(shapeRSigma, cute::GenRowMajor{});
    CHECK_CUDART(cudaMalloc(&DeviceMu, sizeof(Element) * cute::size(layoutMu)));
    CHECK_CUDART(
        cudaMalloc(&DeviceRSigma, sizeof(Element) * cute::size(layoutRSigma)));
    CHECK_CUDART(
        cudaMemset(DeviceMu, 0, sizeof(Element) * cute::size(layoutMu)));
    CHECK_CUDART(cudaMemset(DeviceRSigma, 0,
                            sizeof(Element) * cute::size(layoutRSigma)));

    CHECK_CUDART(cudaDeviceSynchronize());

    tensorMu = std::make_shared<dllm::Tensor1D>(
        DeviceMu, layoutMu, dllm::toDtype<Element>(), dllm::CUDA);
    tensorRSigma = std::make_shared<dllm::Tensor1D>(
        DeviceRSigma, layoutRSigma, dllm::toDtype<Element>(), dllm::CUDA);
  }

  void TearDown() override {
    CHECK_CUDART(cudaFree(DeviceMu));
    CHECK_CUDART(cudaFree(DeviceRSigma));
    CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
  }

  void TestForwardRoutine();

  void TestBackwardRoutine();
};

void TestFlashAttnLayerNorm::TestForwardRoutine() {
  void *DeviceZ, *DeviceX0, *DeviceGamma, *DeviceBeta;
  auto shapeInput = cute::make_shape(T, d);
  auto shapeWeight = cute::make_shape(d);
  auto shapeBias = cute::make_shape(d);

  auto layoutZ = cute::make_layout(shapeInput, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&DeviceZ, sizeof(Element) * cute::size(layoutZ)));

  CHECK_CUDART(cudaMemset(DeviceZ, 0, sizeof(Element) * cute::size(layoutZ)));

  auto tensorZ = std::make_shared<dllm::Tensor2D>(
      DeviceZ, layoutZ, dllm::toDtype<Element>(), dllm::CUDA);

  auto layoutX0 = cute::make_layout(shapeInput, cute::GenRowMajor{});
  auto layoutGamma = cute::make_layout(shapeWeight, cute::GenRowMajor{});
  auto layoutBeta = cute::make_layout(shapeBias, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&DeviceX0, sizeof(Element) * cute::size(layoutX0)));
  CHECK_CUDART(
      cudaMalloc(&DeviceGamma, sizeof(Element) * cute::size(layoutGamma)));
  CHECK_CUDART(
      cudaMalloc(&DeviceBeta, sizeof(Element) * cute::size(layoutBeta)));

  CHECK_CUDART(cudaMemcpy(
      DeviceX0, input.data_ptr<typename TypeToTorch<Element>::Type>(),
      sizeof(Element) * cute::size(layoutX0), cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(
      DeviceGamma, weight.data_ptr<typename TypeToTorch<Element>::Type>(),
      sizeof(Element) * cute::size(layoutGamma), cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(
      DeviceBeta, bias.data_ptr<typename TypeToTorch<Element>::Type>(),
      sizeof(Element) * cute::size(layoutBeta), cudaMemcpyHostToDevice));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorX0 = std::make_shared<dllm::Tensor2D>(
      DeviceX0, layoutX0, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorGamma = std::make_shared<dllm::Tensor1D>(
      DeviceGamma, layoutGamma, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorBeta = std::make_shared<dllm::Tensor1D>(
      DeviceBeta, layoutBeta, dllm::toDtype<Element>(), dllm::CUDA);

  auto tast = dllm::flash_attn::layer_norm::forward(
      tensorZ, tensorMu, tensorRSigma, tensorX0, tensorGamma, tensorBeta, eps);
  tast(&context);

  auto output2 = torch::empty_like(output1);
  auto weight2 = torch::empty_like(weight1);
  auto bias2 = torch::empty_like(bias1);

  CHECK_CUDART(cudaMemcpy(
      output2.data_ptr<typename TypeToTorch<Element>::Type>(), DeviceZ,
      sizeof(Element) * cute::size(layoutZ), cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(
      weight2.data_ptr<typename TypeToTorch<Element>::Type>(), DeviceGamma,
      sizeof(Element) * cute::size(layoutGamma), cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(bias2.data_ptr<typename TypeToTorch<Element>::Type>(),
                          DeviceBeta, sizeof(Element) * cute::size(layoutBeta),
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
  CHECK_CUDART(cudaFree(DeviceX0));
  CHECK_CUDART(cudaFree(DeviceGamma));
  CHECK_CUDART(cudaFree(DeviceBeta));
}

void TestFlashAttnLayerNorm::TestBackwardRoutine() {
  void *DeviceDz, *DeviceX0, *DeviceGamma;
  void *DeviceDx0, *DeviceDgamma, *DeviceDbeta;

  auto shapeInput = cute::make_shape(T, d);
  auto shapeWeight = cute::make_shape(d);
  auto shapeBias = cute::make_shape(d);

  auto layoutDz = cute::make_layout(shapeInput, cute::GenRowMajor{});
  auto layoutX0 = cute::make_layout(shapeInput, cute::GenRowMajor{});
  auto layoutGamma = cute::make_layout(shapeWeight, cute::GenRowMajor{});
  auto layoutDx0 = cute::make_layout(shapeInput, cute::GenRowMajor{});
  auto layoutDgamma = cute::make_layout(shapeWeight, cute::GenRowMajor{});
  auto layoutDbeta = cute::make_layout(shapeBias, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&DeviceDz, sizeof(Element) * cute::size(layoutDz)));
  CHECK_CUDART(cudaMalloc(&DeviceX0, sizeof(Element) * cute::size(layoutX0)));
  CHECK_CUDART(
      cudaMalloc(&DeviceGamma, sizeof(Element) * cute::size(layoutGamma)));
  CHECK_CUDART(cudaMalloc(&DeviceDx0, sizeof(Element) * cute::size(layoutDx0)));
  CHECK_CUDART(
      cudaMalloc(&DeviceDgamma, sizeof(Element) * cute::size(layoutDgamma)));
  CHECK_CUDART(
      cudaMalloc(&DeviceDbeta, sizeof(Element) * cute::size(layoutDbeta)));

  CHECK_CUDART(cudaMemcpy(
      DeviceDz, grad_output.data_ptr<typename TypeToTorch<Element>::Type>(),
      sizeof(Element) * cute::size(layoutDz), cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(
      DeviceX0, input.data_ptr<typename TypeToTorch<Element>::Type>(),
      sizeof(Element) * cute::size(layoutX0), cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(
      DeviceGamma, weight.data_ptr<typename TypeToTorch<Element>::Type>(),
      sizeof(Element) * cute::size(layoutGamma), cudaMemcpyHostToDevice));
  CHECK_CUDART(
      cudaMemset(DeviceDx0, 0, sizeof(Element) * cute::size(layoutDx0)));
  CHECK_CUDART(
      cudaMemset(DeviceDgamma, 0, sizeof(Element) * cute::size(layoutDgamma)));
  CHECK_CUDART(
      cudaMemset(DeviceDbeta, 0, sizeof(Element) * cute::size(layoutDbeta)));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorDz = std::make_shared<dllm::Tensor2D>(
      DeviceDz, layoutDz, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorX0 = std::make_shared<dllm::Tensor2D>(
      DeviceX0, layoutX0, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorGamma = std::make_shared<dllm::Tensor1D>(
      DeviceGamma, layoutGamma, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorDx0 = std::make_shared<dllm::Tensor2D>(
      DeviceDx0, layoutDx0, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorDgamma = std::make_shared<dllm::Tensor1D>(
      DeviceDgamma, layoutDgamma, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorDbeta = std::make_shared<dllm::Tensor1D>(
      DeviceDbeta, layoutDbeta, dllm::toDtype<Element>(), dllm::CUDA);

  auto tast = dllm::flash_attn::layer_norm::backward(
      tensorDz, tensorX0, tensorMu, tensorRSigma, tensorGamma, tensorDx0,
      tensorDgamma, tensorDbeta);
  tast(&context);

  auto grad_input2 = torch::empty_like(grad_input1);
  auto grad_weight2 = torch::empty_like(grad_weight1);
  auto grad_bias2 = torch::empty_like(grad_bias1);

  CHECK_CUDART(cudaMemcpy(
      grad_input2.data_ptr<typename TypeToTorch<Element>::Type>(), DeviceDx0,
      sizeof(Element) * cute::size(layoutDx0), cudaMemcpyDeviceToHost));
  CHECK_CUDART(
      cudaMemcpy(grad_weight2.data_ptr<typename TypeToTorch<Element>::Type>(),
                 DeviceDgamma, sizeof(Element) * cute::size(layoutDgamma),
                 cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(
      grad_bias2.data_ptr<typename TypeToTorch<Element>::Type>(), DeviceDbeta,
      sizeof(Element) * cute::size(layoutDbeta), cudaMemcpyDeviceToHost));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto isApprox_grad_input = grad_input1.allclose(grad_input2, tol);
  auto isApprox_grad_weight = grad_weight1.allclose(grad_weight2, tol);
  auto isApprox_grad_bias = grad_bias1.allclose(grad_bias2, tol);

  if (!isApprox_grad_input) {
    std::ofstream file_grad_input_ref("grad_input1.txt");
    file_grad_input_ref << grad_input1 << std::endl;
    file_grad_input_ref.close();
    std::ofstream file_grad_input("grad_input2.txt");
    file_grad_input << grad_input2 << std::endl;
    file_grad_input.close();
  }

  if (!isApprox_grad_weight) {
    std::ofstream file_grad_weight_ref("grad_weight1.txt");
    file_grad_weight_ref << grad_weight1 << std::endl;
    file_grad_weight_ref.close();
    std::ofstream file_grad_weight("grad_weight2.txt");
    file_grad_weight << grad_weight2 << std::endl;
    file_grad_weight.close();
  }

  if (!isApprox_grad_bias) {
    std::ofstream file_grad_bias_ref("grad_bias1.txt");
    file_grad_bias_ref << grad_bias1 << std::endl;
    file_grad_bias_ref.close();
    std::ofstream file_grad_bias("grad_bias2.txt");
    file_grad_bias << grad_bias2 << std::endl;
    file_grad_bias.close();
  }
  CHECK_CUDART(cudaFree(DeviceDz));
  CHECK_CUDART(cudaFree(DeviceX0));
  CHECK_CUDART(cudaFree(DeviceGamma));
  CHECK_CUDART(cudaFree(DeviceDx0));
  CHECK_CUDART(cudaFree(DeviceDgamma));
  CHECK_CUDART(cudaFree(DeviceDbeta));
}

TEST_F(TestFlashAttnLayerNorm, TestFloat) {
  TestForwardRoutine();
  TestBackwardRoutine();
}
