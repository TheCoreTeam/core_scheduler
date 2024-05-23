#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <fstream>

#include "compute/add.h"

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

template <typename Element>
class TestDLLMResidual : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};

  const dllm::TensorIndexType M = 32;
  const dllm::TensorIndexType N = 64;
  const dllm::TensorIndexType K = 128;

  torch::Device device = torch::kCPU;
  torch::Dtype dtype = TypeTrait<Element>::dtype;

  at::Tensor A;
  at::Tensor B;
  at::Tensor output;

  at::Tensor grad_output;
  at::Tensor grad_A;
  at::Tensor grad_B;

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

  torch::manual_seed(2024);

  A = torch::rand(
      {M, N, K},
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));

  B = torch::rand(
      {M, N, K},
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));

  output = A + B;

  grad_output = torch::rand_like(A);
  output.backward(grad_output, true);

  grad_A = A.grad();
  grad_B = B.grad();

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
  CHECK_CUDART(cudaMemcpy(d_input, A.data_ptr(),
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyHostToDevice));

  CHECK_CUDART(cudaMalloc(&d_residual, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaMemcpy(d_residual, B.data_ptr(),
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

  auto task_forward =
      dllm::compute::Add::forward(tensor_output, tensor_input, tensor_residual);
  task_forward(&context);

  auto my_output = torch::empty_like(A);

  CHECK_CUDART(cudaMemcpy(my_output.data_ptr(), d_output,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto is_approx_output = output.allclose(my_output);

  ASSERT_TRUE(is_approx_output);

  if (!is_approx_output) {
    std::ofstream file_output_ref("output" + TypeTrait<Element>::label +
                                  ".txt");
    file_output_ref << output << std::endl;
    file_output_ref.close();
    std::ofstream file_output("my_output" + TypeTrait<Element>::label + ".txt");
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

  auto task_backward = dllm::compute::Add::backward(
      tensor_grad_input, tensor_grad_residual, tensor_grad_output);
  task_backward(&context);

  auto my_grad_input = torch::empty_like(A);
  auto my_grad_residual = torch::empty_like(A);

  CHECK_CUDART(cudaMemcpy(my_grad_input.data_ptr(), d_grad_input,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(my_grad_residual.data_ptr(), d_grad_residual,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto is_approx_grad_input = grad_A.allclose(my_grad_input);
  auto is_approx_grad_residual = grad_B.allclose(my_grad_residual);

  if (!is_approx_grad_input) {
    std::ofstream file_weight_ref("grad_input" + TypeTrait<Element>::label +
                                  ".txt");
    file_weight_ref << grad_A << std::endl;
    file_weight_ref.close();
    std::ofstream file_weight("my_grad_input" + TypeTrait<Element>::label +
                              ".txt");
    file_weight << my_grad_input << std::endl;
    file_weight.close();
  }

  if (!is_approx_grad_residual) {
    std::ofstream file_bias_ref("grad_residual" + TypeTrait<Element>::label +
                                ".txt");
    file_bias_ref << grad_B << std::endl;
    file_bias_ref.close();
    std::ofstream file_bias("my_grad_residual" + TypeTrait<Element>::label +
                            ".txt");
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
