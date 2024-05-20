#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <fstream>

#include "compute/residual.h"
#include "logger.h"
#include "tensor.h"

template <typename T>
struct TypeToTorch;

template <>
struct TypeToTorch<float> {
  static constexpr at::ScalarType dtype = torch::kFloat;
};

template <>
struct TypeToTorch<nv_half> {
  static constexpr at::ScalarType dtype = torch::kHalf;
};

template <>
struct TypeToTorch<double> {
  static constexpr at::ScalarType dtype = torch::kDouble;
};

template <typename Element>
class TestDLLMResidual : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};

  const dllm::TensorIndexType M = 32;
  const dllm::TensorIndexType N = 64;
  const dllm::TensorIndexType K = 128;

  torch::Device device = torch::kCPU;
  torch::Dtype dtype = TypeToTorch<Element>::dtype;

  at::Tensor input;
  at::Tensor residual;
  at::Tensor output;

  at::Tensor grad_output;
  at::Tensor grad_input;
  at::Tensor grad_residual;

  void *d_grad_output;
  std::shared_ptr<dllm::Tensor3D> tensor_grad_output;

  void SetUp() override;
  void TearDown() override;

  void TestForwardRoutine();
  void TestBackwardRoutine();
};

template <typename Element>
void TestDLLMResidual<Element>::SetUp() {
  CHECK_CUDART(
      cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
  //    CHECK_CUDART(cudaDeviceGetDefaultMemPool(&context.memPool, 0));
  torch::manual_seed(2024);

  input = torch::rand(
      {M, N, K},
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));

  residual = torch::rand(
      {M, N, K},
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));

  output = input + residual;

  grad_output = torch::rand_like(input);
  output.backward(grad_output, true);

  grad_input = input.grad();
  grad_residual = residual.grad();

  auto shape = cute::make_shape(M, N, K);
  auto layout = cute::make_layout(shape, cute::GenRowMajor{});

  CHECK_CUDART(
      cudaMalloc(&d_grad_output, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaMemcpy(d_grad_output, grad_output.data_ptr(),
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  tensor_grad_output = std::make_shared<dllm::Tensor3D>(
      d_grad_output, layout, dllm::toDtype<Element>(), dllm::CUDA);
}

template <typename Element>
void TestDLLMResidual<Element>::TearDown() {
  CHECK_CUDART(cudaFree(d_grad_output));
  CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
}

template <typename Element>
void TestDLLMResidual<Element>::TestForwardRoutine() {
  auto shape = cute::make_shape(M, N, K);
  auto layout = cute::make_layout(shape, cute::GenRowMajor{});

  void *d_input, *d_residual, *d_output;

  CHECK_CUDART(cudaMalloc(&d_input, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaMemcpy(d_input, input.data_ptr(),
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyHostToDevice));

  CHECK_CUDART(cudaMalloc(&d_residual, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaMemcpy(d_residual, residual.data_ptr(),
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyHostToDevice));

  CHECK_CUDART(cudaMalloc(&d_output, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaMemset(d_output, 0, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensor_input = std::make_shared<dllm::Tensor3D>(
      d_input, layout, dllm::toDtype<Element>(), dllm::CUDA);

  auto tensor_residual = std::make_shared<dllm::Tensor3D>(
      d_residual, layout, dllm::toDtype<Element>(), dllm::CUDA);

  auto tensor_output = std::make_shared<dllm::Tensor3D>(
      d_output, layout, dllm::toDtype<Element>(), dllm::CUDA);

  auto task_forward = dllm::compute::Residual::forward(
      tensor_input, tensor_residual, tensor_output);
  task_forward(&context);

  auto my_output = torch::empty_like(input);

  CHECK_CUDART(cudaMemcpy(my_output.data_ptr(), d_output,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto is_approx_output = output.allclose(my_output);

  ASSERT_TRUE(is_approx_output);

  if (!is_approx_output) {
    std::ofstream file_output_ref("output.txt");
    file_output_ref << output << std::endl;
    file_output_ref.close();
    std::ofstream file_output("my_output.txt");
    file_output << my_output << std::endl;
    file_output.close();
  }

  CHECK_CUDART(cudaFree(d_input));
  CHECK_CUDART(cudaFree(d_residual));
  CHECK_CUDART(cudaFree(d_output));
}

template <typename Element>
void TestDLLMResidual<Element>::TestBackwardRoutine() {
  auto shape = cute::make_shape(M, N, K);
  auto layout = cute::make_layout(shape, cute::GenRowMajor{});
  void *d_grad_input, *d_grad_residual;

  CHECK_CUDART(cudaMalloc(&d_grad_input, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(
      cudaMemset(d_grad_input, 0, sizeof(Element) * cute::size(layout)));

  CHECK_CUDART(
      cudaMalloc(&d_grad_residual, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(
      cudaMemset(d_grad_residual, 0, sizeof(Element) * cute::size(layout)));

  CHECK_CUDART(cudaDeviceSynchronize());
  auto tensor_grad_input = std::make_shared<dllm::Tensor3D>(
      d_grad_input, layout, dllm::toDtype<Element>(), dllm::CUDA);

  auto tensor_grad_residual = std::make_shared<dllm::Tensor3D>(
      d_grad_residual, layout, dllm::toDtype<Element>(), dllm::CUDA);

  auto task_backward = dllm::compute::Residual::backward(
      tensor_grad_output, tensor_grad_input, tensor_grad_residual);
  task_backward(&context);

  auto my_grad_input = torch::empty_like(input);
  auto my_grad_residual = torch::empty_like(input);

  CHECK_CUDART(cudaMemcpy(my_grad_input.data_ptr(), d_grad_input,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(my_grad_residual.data_ptr(), d_grad_residual,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto is_approx_grad_input = grad_input.allclose(my_grad_input);
  auto is_approx_grad_residual = grad_residual.allclose(my_grad_residual);

  if (!is_approx_grad_input) {
    std::ofstream file_weight_ref("grad_input.txt");
    file_weight_ref << grad_input << std::endl;
    file_weight_ref.close();
    std::ofstream file_weight("my_grad_input.txt");
    file_weight << my_grad_input << std::endl;
    file_weight.close();
  }

  if (!is_approx_grad_residual) {
    std::ofstream file_bias_ref("grad_residual.txt");
    file_bias_ref << grad_residual << std::endl;
    file_bias_ref.close();
    std::ofstream file_bias("my_grad_residual.txt");
    file_bias << my_grad_residual << std::endl;
    file_bias.close();
  }

  ASSERT_TRUE(is_approx_grad_input);
  ASSERT_TRUE(is_approx_grad_residual);
  CHECK_CUDART(cudaFree(d_grad_input));
  CHECK_CUDART(cudaFree(d_grad_residual));
}

using ElementTypes = ::testing::Types<float, nv_half, double>;
TYPED_TEST_SUITE(TestDLLMResidual, ElementTypes);
TYPED_TEST(TestDLLMResidual, TestVaryingElementTypes) {
  this->TestForwardRoutine();
  this->TestBackwardRoutine();
}
