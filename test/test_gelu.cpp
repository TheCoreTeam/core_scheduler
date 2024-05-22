#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <math_constants.h>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <torch/torch.h>
#include "compute/gelu.h"
#include "logger.h"
#include "tensor.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

template<typename T>
struct TypeToTorch;

template<>
struct TypeToTorch<float> {
  using Type = float;
  static const at::ScalarType type = torch::kFloat;
};

template<>
struct TypeToTorch<nv_half> {
  using Type = c10::Half;
  static const at::ScalarType type = torch::kHalf;
};

template<>
struct TypeToTorch<double> {
  using Type = double;
  static const at::ScalarType type = torch::kDouble;
};

namespace Eigen::internal {
template <>
struct scalar_random_op<nv_half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_random_op);
  inline const nv_half operator()() const {
    return static_cast<nv_half>(random<float>());
  }
};
}  // namespace Eigen::internal

class TestDLLMGelu : public ::testing::Test {
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
  void TestRoutine(const dllm::TensorIndexType T, const double tol_forward, const double tol_backward);
};

template <typename Element>
void TestDLLMGelu::TestRoutine(const dllm::TensorIndexType T, const double tol_forward, const double tol_backward) {

  const dllm::TensorIndexType B = 2;
  torch::Device device = torch::kCPU;
  torch::Dtype dtype = TypeToTorch<Element>::type;
  auto input = torch::randn({B, T}, torch::TensorOptions().dtype(dtype).device(device));

  auto shape = cute::make_shape(B, T);
  auto layout = cute::make_layout(shape, cute::GenRowMajor{});


  auto input1 = input.detach().clone().set_requires_grad(true);

  auto input2 = input.detach().clone();


  void *DeviceInput, *DeviceOutput;
  CHECK_CUDART(cudaMalloc(&DeviceInput, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaMalloc(&DeviceOutput, sizeof(Element) * cute::size(layout)));
  auto tensorInput = std::make_shared<dllm::Tensor2D>(
      DeviceInput, layout, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorOutput = std::make_shared<dllm::Tensor2D>(
      DeviceOutput, layout, dllm::toDtype<Element>(), dllm::CUDA);

  CHECK_CUDART(cudaMemcpy(DeviceInput, input2.data_ptr<typename TypeToTorch<Element>::Type>(),
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(
      cudaMemset(DeviceOutput, 0, sizeof(Element) * cute::size(layout)));

  CHECK_CUDART(cudaDeviceSynchronize());

  // 应用GELU激活函数
  auto Output1 = torch::gelu(input1);

  auto task = dllm::compute::GeLU::forward(tensorOutput, tensorInput);
  task(&context);

  auto Output2 = torch::empty_like(Output1, torch::TensorOptions().dtype(dtype));

  CHECK_CUDART(cudaMemcpy(Output2.data_ptr<typename TypeToTorch<Element>::Type>(), DeviceOutput,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto isApprox_forward = Output2.allclose(Output1, tol_forward);

  ASSERT_TRUE(isApprox_forward);


  if (!isApprox_forward) {
    std::ofstream fileOuput("Output1.txt");
    fileOuput << Output1 << std::endl;
    fileOuput.close();
    std::ofstream fileRef("Output2.txt");
    fileRef << Output2 << std::endl;
    fileRef.close();
  }

  auto GradOutput = torch::randn_like(Output1);

  void *DeviceGradInput, *DeviceGradOutput;
  CHECK_CUDART(cudaMalloc(&DeviceGradInput, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaMalloc(&DeviceGradOutput, sizeof(Element) * cute::size(layout)));
  auto tensorGradInput = std::make_shared<dllm::Tensor2D>(
      DeviceGradInput, layout, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorGradOutput = std::make_shared<dllm::Tensor2D>(
      DeviceGradOutput, layout, dllm::toDtype<Element>(), dllm::CUDA);

  CHECK_CUDART(cudaMemcpy(DeviceGradOutput, GradOutput.data_ptr<typename TypeToTorch<Element>::Type>(),
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(
      cudaMemset(DeviceGradInput, 0, sizeof(Element) * cute::size(layout)));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto GradInput1 = torch::autograd::grad({Output1}, {input1}, {GradOutput}, /*retain_graph=*/false, /*create_graph=*/false, /*allow_unused=*/true)[0];

  auto task_backward = dllm::compute::GeLU::backward(tensorGradInput, tensorInput, tensorGradOutput);
  task_backward(&context);

  auto GradInput2 = torch::empty_like(GradInput1, torch::TensorOptions().dtype(dtype));

  CHECK_CUDART(cudaMemcpy(GradInput2.data_ptr<typename TypeToTorch<Element>::Type>(), DeviceGradInput,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto isApprox_backward = GradInput2.allclose(GradInput1,tol_backward);



  if (!isApprox_backward) {
    std::ofstream fileOuput("GradInput1.txt");
    fileOuput << GradInput1 << std::endl;
    fileOuput.close();
    std::ofstream fileRef("GradInput2.txt");
    fileRef << GradInput2 << std::endl;
    fileRef.close();
  }

//  ASSERT_TRUE(isApprox_backward);
//  for (int i = 0; i < B * T; ++i) {
//    auto close = at::all_close(GradInput2[i], GradInput1[i],
//                  tol_backward * (GradInput1[i].abs()));
//  }

  // 计算两个张量之间的绝对误差
  at::Tensor abs_error = torch::abs(GradInput1 - GradInput2);

  // 检查每个元素的绝对误差是否小于tol
  bool all_less_than_tol = (abs_error <= tol_backward).all().item<bool>();

  // 使用ASSERT_TRUE来判断所有元素的误差是否都小于tol
  ASSERT_TRUE(all_less_than_tol);


  CHECK_CUDART(cudaFree(DeviceInput));
  CHECK_CUDART(cudaFree(DeviceOutput));
  CHECK_CUDART(cudaFree(DeviceGradInput));
  CHECK_CUDART(cudaFree(DeviceGradOutput));
}

TEST_F(TestDLLMGelu, TestF32_128) { TestRoutine<float>(128, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF32_512) { TestRoutine<float>(512, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF32_1024) { TestRoutine<float>(1024, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF16_128) { TestRoutine<nv_half>(128, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF16_512) { TestRoutine<nv_half>(512, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF16_1024) { TestRoutine<nv_half>(1024, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF64_128) { TestRoutine<double>(128, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF64_512) { TestRoutine<double>(512, 1e-2, 1e-2); }

TEST_F(TestDLLMGelu, TestF64_1024) { TestRoutine<double>(1024, 1e-2, 1e-2); }