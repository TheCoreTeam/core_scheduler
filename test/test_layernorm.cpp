#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <flash_attention/layer_norm/layer_norm.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>

template <typename T>
struct TypeTrait;

template <>
struct TypeTrait<float> {
  static constexpr at::ScalarType dtype = torch::kFloat;
  static constexpr std::string label = "float";
};

template <>
struct TypeTrait<nv_half> {
  static constexpr at::ScalarType dtype = torch::kHalf;
  static constexpr std::string label = "half";
};

template <>
struct TypeTrait<double> {
  static constexpr at::ScalarType dtype = torch::kDouble;
  static constexpr std::string label = "double";
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
  torch::Dtype dtype = TypeTrait<Element>::dtype;

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

  void *device_mu, *device_rsigma;
  std::shared_ptr<dllm::Tensor1D> tensor_mu, tensor_rsigma;

  void SetUp() override;

  void TearDown() override;

  void TestForwardRoutine();

  void TestBackwardRoutine();
};

void TestFlashAttnLayerNorm::SetUp() {
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  CHECK_CUDART(cudaDeviceGetDefaultMemPool(&context.memPool, 0));
  torch::manual_seed(2024);

  weight = torch::randn(d, torch::TensorOptions().dtype(dtype).device(device));
  bias = torch::randn(d, torch::TensorOptions().dtype(dtype).device(device));
  input =
      torch::randn({T, d}, torch::TensorOptions().dtype(dtype).device(device));

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

  auto shape_mu = cute::make_shape(T);
  auto shape_rsigma = cute::make_shape(T);
  auto layout_mu = cute::make_layout(shape_mu, cute::GenRowMajor{});
  auto layout_rsigma = cute::make_layout(shape_rsigma, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&device_mu, sizeof(Element) * cute::size(layout_mu)));
  CHECK_CUDART(
      cudaMalloc(&device_rsigma, sizeof(Element) * cute::size(layout_rsigma)));
  CHECK_CUDART(
      cudaMemset(device_mu, 0, sizeof(Element) * cute::size(layout_mu)));
  CHECK_CUDART(cudaMemset(device_rsigma, 0,
                          sizeof(Element) * cute::size(layout_rsigma)));

  CHECK_CUDART(cudaDeviceSynchronize());

  tensor_mu = std::make_shared<dllm::Tensor1D>(
      device_mu, layout_mu, dllm::toDtype<Element>(), dllm::CUDA);
  tensor_rsigma = std::make_shared<dllm::Tensor1D>(
      device_rsigma, layout_rsigma, dllm::toDtype<Element>(), dllm::CUDA);
}

void TestFlashAttnLayerNorm::TearDown() {
  CHECK_CUDART(cudaFree(device_mu));
  CHECK_CUDART(cudaFree(device_rsigma));
  CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
}

void TestFlashAttnLayerNorm::TestForwardRoutine() {
  void *device_z, *device_x0, *device_gamma, *device_beta;
  auto shape_input = cute::make_shape(T, d);
  auto shape_weight = cute::make_shape(d);
  auto shape_bias = cute::make_shape(d);

  auto layout_z = cute::make_layout(shape_input, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&device_z, sizeof(Element) * cute::size(layout_z)));

  CHECK_CUDART(cudaMemset(device_z, 0, sizeof(Element) * cute::size(layout_z)));

  auto tensor_z = std::make_shared<dllm::Tensor2D>(
      device_z, layout_z, dllm::toDtype<Element>(), dllm::CUDA);

  auto layout_x0 = cute::make_layout(shape_input, cute::GenRowMajor{});
  auto layout_gamma = cute::make_layout(shape_weight, cute::GenRowMajor{});
  auto layout_beta = cute::make_layout(shape_bias, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&device_x0, sizeof(Element) * cute::size(layout_x0)));
  CHECK_CUDART(
      cudaMalloc(&device_gamma, sizeof(Element) * cute::size(layout_gamma)));
  CHECK_CUDART(
      cudaMalloc(&device_beta, sizeof(Element) * cute::size(layout_beta)));

  CHECK_CUDART(cudaMemcpy(device_x0, input.data_ptr(),
                          sizeof(Element) * cute::size(layout_x0),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(device_gamma, weight.data_ptr(),
                          sizeof(Element) * cute::size(layout_gamma),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(device_beta, bias.data_ptr(),
                          sizeof(Element) * cute::size(layout_beta),
                          cudaMemcpyHostToDevice));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensor_x0 = std::make_shared<dllm::Tensor2D>(
      device_x0, layout_x0, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensor_gamma = std::make_shared<dllm::Tensor1D>(
      device_gamma, layout_gamma, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensor_beta = std::make_shared<dllm::Tensor1D>(
      device_beta, layout_beta, dllm::toDtype<Element>(), dllm::CUDA);

  auto task = dllm::flash_attn::LayerNorm::forward(
      tensor_z, tensor_mu, tensor_rsigma, tensor_x0, tensor_gamma, tensor_beta,
      eps);
  task(&context);

  auto output2 = torch::empty_like(output1);
  auto weight2 = torch::empty_like(weight1);
  auto bias2 = torch::empty_like(bias1);

  CHECK_CUDART(cudaMemcpy(output2.data_ptr(), device_z,
                          sizeof(Element) * cute::size(layout_z),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(weight2.data_ptr(), device_gamma,
                          sizeof(Element) * cute::size(layout_gamma),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(bias2.data_ptr(), device_beta,
                          sizeof(Element) * cute::size(layout_beta),
                          cudaMemcpyDeviceToHost));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto is_approx_input = output1.allclose(output2, tol);
  auto is_approx_weight = weight1.allclose(weight2, tol);
  auto is_approx_bias = bias1.allclose(bias2, tol);

  if (!is_approx_input) {
    std::ofstream file_output_ref("output1" + TypeTrait<Element>::label +
                                  ".txt");
    file_output_ref << output1 << std::endl;
    file_output_ref.close();
    std::ofstream file_output("output2" + TypeTrait<Element>::label + ".txt");
    file_output << output2 << std::endl;
    file_output.close();
  }

  if (!is_approx_weight) {
    std::ofstream file_weight_ref("weight1" + TypeTrait<Element>::label +
                                  ".txt");
    file_weight_ref << weight1 << std::endl;
    file_weight_ref.close();
    std::ofstream file_weight("weight2" + TypeTrait<Element>::label + ".txt");
    file_weight << weight2 << std::endl;
    file_weight.close();
  }

  if (!is_approx_bias) {
    std::ofstream file_bias_ref("bias1" + TypeTrait<Element>::label + ".txt");
    file_bias_ref << bias1 << std::endl;
    file_bias_ref.close();
    std::ofstream file_bias("bias2" + TypeTrait<Element>::label + ".txt");
    file_bias << bias2 << std::endl;
    file_bias.close();
  }

  ASSERT_TRUE(is_approx_input);
  ASSERT_TRUE(is_approx_weight);
  ASSERT_TRUE(is_approx_bias);

  CHECK_CUDART(cudaFree(device_z));
  CHECK_CUDART(cudaFree(device_x0));
  CHECK_CUDART(cudaFree(device_gamma));
  CHECK_CUDART(cudaFree(device_beta));
}

void TestFlashAttnLayerNorm::TestBackwardRoutine() {
  void *device_dz, *device_x0, *device_gamma;
  void *device_dx0, *device_dgamma, *device_dbeta;

  auto shape_input = cute::make_shape(T, d);
  auto shape_weight = cute::make_shape(d);
  auto shape_bias = cute::make_shape(d);

  auto layout_dz = cute::make_layout(shape_input, cute::GenRowMajor{});
  auto layout_x0 = cute::make_layout(shape_input, cute::GenRowMajor{});
  auto layout_gamma = cute::make_layout(shape_weight, cute::GenRowMajor{});
  auto layout_dx0 = cute::make_layout(shape_input, cute::GenRowMajor{});
  auto layout_dgamma = cute::make_layout(shape_weight, cute::GenRowMajor{});
  auto layout_dbeta = cute::make_layout(shape_bias, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&device_dz, sizeof(Element) * cute::size(layout_dz)));
  CHECK_CUDART(cudaMalloc(&device_x0, sizeof(Element) * cute::size(layout_x0)));
  CHECK_CUDART(
      cudaMalloc(&device_gamma, sizeof(Element) * cute::size(layout_gamma)));
  CHECK_CUDART(
      cudaMalloc(&device_dx0, sizeof(Element) * cute::size(layout_dx0)));
  CHECK_CUDART(
      cudaMalloc(&device_dgamma, sizeof(Element) * cute::size(layout_dgamma)));
  CHECK_CUDART(
      cudaMalloc(&device_dbeta, sizeof(Element) * cute::size(layout_dbeta)));

  CHECK_CUDART(cudaMemcpy(device_dz, grad_output.data_ptr(),
                          sizeof(Element) * cute::size(layout_dz),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(device_x0, input.data_ptr(),
                          sizeof(Element) * cute::size(layout_x0),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(device_gamma, weight.data_ptr(),
                          sizeof(Element) * cute::size(layout_gamma),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(
      cudaMemset(device_dx0, 0, sizeof(Element) * cute::size(layout_dx0)));
  CHECK_CUDART(cudaMemset(device_dgamma, 0,
                          sizeof(Element) * cute::size(layout_dgamma)));
  CHECK_CUDART(
      cudaMemset(device_dbeta, 0, sizeof(Element) * cute::size(layout_dbeta)));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensor_dz = std::make_shared<dllm::Tensor2D>(
      device_dz, layout_dz, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensor_x0 = std::make_shared<dllm::Tensor2D>(
      device_x0, layout_x0, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensor_gamma = std::make_shared<dllm::Tensor1D>(
      device_gamma, layout_gamma, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensor_dx0 = std::make_shared<dllm::Tensor2D>(
      device_dx0, layout_dx0, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensor_dgamma = std::make_shared<dllm::Tensor1D>(
      device_dgamma, layout_dgamma, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensor_dbeta = std::make_shared<dllm::Tensor1D>(
      device_dbeta, layout_dbeta, dllm::toDtype<Element>(), dllm::CUDA);

  auto task = dllm::flash_attn::LayerNorm::backward(
      tensor_dx0, tensor_dgamma, tensor_dbeta, tensor_dz, tensor_x0, tensor_mu,
      tensor_rsigma, tensor_gamma);
  task(&context);

  auto grad_input2 = torch::empty_like(grad_input1);
  auto grad_weight2 = torch::empty_like(grad_weight1);
  auto grad_bias2 = torch::empty_like(grad_bias1);

  CHECK_CUDART(cudaMemcpy(grad_input2.data_ptr(), device_dx0,
                          sizeof(Element) * cute::size(layout_dx0),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(grad_weight2.data_ptr(), device_dgamma,
                          sizeof(Element) * cute::size(layout_dgamma),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(grad_bias2.data_ptr(), device_dbeta,
                          sizeof(Element) * cute::size(layout_dbeta),
                          cudaMemcpyDeviceToHost));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto is_approx_grad_input = grad_input1.allclose(grad_input2, tol);
  auto is_approx_grad_weight = grad_weight1.allclose(grad_weight2, tol);
  auto is_approx_grad_bias = grad_bias1.allclose(grad_bias2, tol);

  if (!is_approx_grad_input) {
    std::ofstream file_grad_input_ref("grad_input1" +
                                      TypeTrait<Element>::label + ".txt");
    file_grad_input_ref << grad_input1 << std::endl;
    file_grad_input_ref.close();
    std::ofstream file_grad_input("grad_input2" + TypeTrait<Element>::label +
                                  ".txt");
    file_grad_input << grad_input2 << std::endl;
    file_grad_input.close();
  }

  if (!is_approx_grad_weight) {
    std::ofstream file_grad_weight_ref("grad_weight1" +
                                       TypeTrait<Element>::label + ".txt");
    file_grad_weight_ref << grad_weight1 << std::endl;
    file_grad_weight_ref.close();
    std::ofstream file_grad_weight("grad_weight2" + TypeTrait<Element>::label +
                                   ".txt");
    file_grad_weight << grad_weight2 << std::endl;
    file_grad_weight.close();
  }

  if (!is_approx_grad_bias) {
    std::ofstream file_grad_bias_ref("grad_bias1" + TypeTrait<Element>::label +
                                     ".txt");
    file_grad_bias_ref << grad_bias1 << std::endl;
    file_grad_bias_ref.close();
    std::ofstream file_grad_bias("grad_bias2" + TypeTrait<Element>::label +
                                 ".txt");
    file_grad_bias << grad_bias2 << std::endl;
    file_grad_bias.close();
  }

  ASSERT_TRUE(is_approx_grad_input);
  ASSERT_TRUE(is_approx_grad_weight);
  ASSERT_TRUE(is_approx_grad_bias);

  CHECK_CUDART(cudaFree(device_dz));
  CHECK_CUDART(cudaFree(device_x0));
  CHECK_CUDART(cudaFree(device_gamma));
  CHECK_CUDART(cudaFree(device_dx0));
  CHECK_CUDART(cudaFree(device_dgamma));
  CHECK_CUDART(cudaFree(device_dbeta));
}

TEST_F(TestFlashAttnLayerNorm, TestFloat) {
  TestForwardRoutine();
  TestBackwardRoutine();
}
