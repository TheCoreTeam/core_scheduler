#include <compute/embedding.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

template <typename T>
struct TypeToTorch;

template <>
struct TypeToTorch<float> {
  using Type = float;
  static const at::ScalarType type = torch::kFloat;
};

template <>
struct TypeToTorch<nv_half> {
  using Type = c10::Half;
  static const at::ScalarType type = torch::kHalf;
};

template <>
struct TypeToTorch<double> {
  using Type = double;
  static const at::ScalarType type = torch::kDouble;
};

namespace Eigen::internal {
template <>
struct scalar_random_op<nv_half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_random_op)
  inline const nv_half operator()() const {
    return static_cast<nv_half>(random<float>());
  }
};
}  // namespace Eigen::internal

class TestEmbedding : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};

  void SetUp() override {
    CHECK_CUDART(
        cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
    CHECK_CUDART(cudaDeviceGetDefaultMemPool(&context.memPool, 0));
  }

  void TearDown() override {
    CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
  }

  template <typename Element>
  void TestRoutine(const double tol_forward, const double tol_backward);
};

template <typename Element>
void TestEmbedding::TestRoutine(const double tol_forward,
                                const double tol_backward) {
  const dllm::TensorIndexType B = 2;
  const dllm::TensorIndexType T = 1024;
  const dllm::TensorIndexType MaxT = 1024;
  const dllm::TensorIndexType vocab = 4095;
  const dllm::TensorIndexType d = 512;
  //  const dllm::TensorIndexType B = 1;
  //  const dllm::TensorIndexType T = 8;
  //  const dllm::TensorIndexType MaxT = 8;
  //  const dllm::TensorIndexType vocab = 16;
  //  const dllm::TensorIndexType d = 8;

  torch::Device device = torch::kCPU;
  torch::Dtype dtype = TypeToTorch<Element>::type;
  auto wte = torch::randn({vocab, d},
                          torch::TensorOptions().dtype(dtype).device(device));
  auto wpe = torch::randn({MaxT, d},
                          torch::TensorOptions().dtype(dtype).device(device));
  auto input = torch::randint(
      0, 3, {B, T}, torch::TensorOptions().dtype(torch::kInt).device(device));
  auto input_wpe =
      torch::range(0, T - 1,
                   torch::TensorOptions().dtype(torch::kInt).device(device))
          .repeat({B, 1});

  auto input1 = input.detach().clone();
  auto wte1 = wte.to(TypeToTorch<Element>::type)
                  .detach()
                  .clone()
                  .set_requires_grad(true);
  auto wpe1 = wpe.to(TypeToTorch<Element>::type)
                  .detach()
                  .clone()
                  .set_requires_grad(true);

  auto input2 = input.detach().clone();
  auto wte2 = wte.detach().clone();
  auto wpe2 = wpe.detach().clone();

  auto output_wte1 = at::embedding(wte1, input1);
  auto output1_wpe1 = at::embedding(wpe1, input_wpe);
  auto output1 = output_wte1 + output1_wpe1;

  void *DeviceInput;
  void *DeviceWte;
  void *DeviceWpe;
  void *DeviceOutput;
  auto shapeInput = cute::make_shape(B, T);
  auto shapeWte = cute::make_shape(vocab, d);
  auto shapeWpe = cute::make_shape(MaxT, d);
  auto shapeOutput = cute::make_shape(B, T, d);
  auto layoutInput = cute::make_layout(shapeInput, cute::GenRowMajor{});
  auto layoutWte = cute::make_layout(shapeWte, cute::GenRowMajor{});
  auto layoutWpe = cute::make_layout(shapeWpe, cute::GenRowMajor{});
  auto layoutOutput = cute::make_layout(shapeOutput, cute::GenRowMajor{});

  CHECK_CUDART(cudaMalloc(&DeviceInput, sizeof(int) * cute::size(layoutInput)));
  CHECK_CUDART(cudaMalloc(&DeviceWte, sizeof(Element) * cute::size(layoutWte)));
  CHECK_CUDART(cudaMalloc(&DeviceWpe, sizeof(Element) * cute::size(layoutWpe)));
  CHECK_CUDART(
      cudaMalloc(&DeviceOutput, sizeof(Element) * cute::size(layoutOutput)));

  CHECK_CUDART(cudaMemcpy(DeviceInput, input2.data_ptr<int>(),
                          sizeof(int) * cute::size(layoutInput),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(
      DeviceWte, wte2.data_ptr<typename TypeToTorch<Element>::Type>(),
      sizeof(Element) * cute::size(layoutWte), cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(
      DeviceWpe, wpe2.data_ptr<typename TypeToTorch<Element>::Type>(),
      sizeof(Element) * cute::size(layoutWpe), cudaMemcpyHostToDevice));

  CHECK_CUDART(
      cudaMemset(DeviceOutput, 0, sizeof(Element) * cute::size(layoutOutput)));

  auto tensorInput = std::make_shared<dllm::Tensor2D>(
      DeviceInput, layoutInput, dllm::toDtype<int>(), dllm::CUDA);
  auto tensorWte = std::make_shared<dllm::Tensor2D>(
      DeviceWte, layoutWte, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorWpe = std::make_shared<dllm::Tensor2D>(
      DeviceWpe, layoutWpe, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorOutput = std::make_shared<dllm::Tensor3D>(
      DeviceOutput, layoutOutput, dllm::toDtype<Element>(), dllm::CUDA);

  CHECK_CUDART(cudaDeviceSynchronize());

  auto task_forward = dllm::compute::embedding::forward(
      tensorOutput, tensorInput, tensorWte, tensorWpe);
  task_forward(&context);

  auto output2 =
      torch::empty_like(output1, torch::TensorOptions().dtype(dtype));

  CHECK_CUDART(cudaMemcpy(
      output2.template data_ptr<typename TypeToTorch<Element>::Type>(),
      DeviceOutput, sizeof(Element) * cute::size(layoutOutput),
      cudaMemcpyDeviceToHost));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto isApprox_output =
      output1.allclose(output2.to(TypeToTorch<Element>::type), tol_forward);

  if (!isApprox_output) {
    std::ofstream fileOuput("output1.txt");
    fileOuput << output1 << std::endl;
    fileOuput.close();
    std::ofstream fileRef("output2.txt");
    fileRef << output2 << std::endl;
    fileRef.close();
  }

  ASSERT_TRUE(isApprox_output);

  // backward check
  auto grad_output = torch::rand_like(output1);

  // 计算梯度
  auto grads = torch::autograd::grad(
      {output1}, {wte1, wpe1}, {grad_output}, /*retain_graph=*/false,
      /*create_graph=*/false, /*allow_unused=*/true);

  // Access and print gradients
  auto grad_wte1 = grads[0];
  auto grad_wpe1 = grads[1];

  void *Device_grad_output;
  void *Device_grad_wte;
  void *Device_grad_wpe;
  auto shapeGradOutput = shapeOutput;
  auto shapeGradWte = shapeWte;
  auto shapeGradWpe = shapeWpe;
  auto layoutGradOutput =
      cute::make_layout(shapeGradOutput, cute::GenRowMajor{});
  auto layoutGradWte = cute::make_layout(shapeGradWte, cute::GenRowMajor{});
  auto layoutGradWpe = cute::make_layout(shapeGradWpe, cute::GenRowMajor{});
  CHECK_CUDART(cudaMalloc(&Device_grad_output,
                          sizeof(Element) * cute::size(layoutGradOutput)));
  CHECK_CUDART(cudaMalloc(&Device_grad_wte,
                          sizeof(Element) * cute::size(layoutGradWte)));
  CHECK_CUDART(
      cudaMalloc(&Device_grad_wpe, sizeof(Element) * cute::size(shapeGradWpe)));
  CHECK_CUDART(cudaMemcpy(
      Device_grad_output,
      grad_output.to(dtype)
          .template data_ptr<typename TypeToTorch<Element>::Type>(),
      sizeof(Element) * cute::size(layoutGradOutput), cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemset(Device_grad_wte, 0,
                          sizeof(Element) * cute::size(layoutGradWte)));
  CHECK_CUDART(cudaMemset(Device_grad_wpe, 0,
                          sizeof(Element) * cute::size(layoutGradWpe)));
  auto tensorGradOutput = std::make_shared<dllm::Tensor3D>(
      Device_grad_output, layoutOutput, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorGradWte = std::make_shared<dllm::Tensor2D>(
      Device_grad_wte, layoutGradWte, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorGradWpe = std::make_shared<dllm::Tensor2D>(
      Device_grad_wpe, layoutGradWpe, dllm::toDtype<Element>(), dllm::CUDA);

  CHECK_CUDART(cudaDeviceSynchronize());

  auto task_backward = dllm::compute::embedding::backward(
      tensorInput, tensorGradWte, tensorGradWpe, tensorGradOutput);
  task_backward(&context);

  auto grad_wte2 =
      torch::empty_like(grad_wte1, torch::TensorOptions().dtype(dtype));
  auto grad_wpe2 =
      torch::empty_like(grad_wpe1, torch::TensorOptions().dtype(dtype));

  CHECK_CUDART(cudaMemcpy(
      grad_wte2.template data_ptr<typename TypeToTorch<Element>::Type>(),
      Device_grad_wte, sizeof(Element) * cute::size(layoutGradWte),
      cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaMemcpy(
      grad_wpe2.template data_ptr<typename TypeToTorch<Element>::Type>(),
      Device_grad_wpe, sizeof(Element) * cute::size(layoutGradWpe),
      cudaMemcpyDeviceToHost));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto isApprox_grad_wte = grad_wte1.allclose(
      grad_wte2.to(TypeToTorch<Element>::type), tol_backward);
  auto isApprox_grad_wpe = grad_wpe1.allclose(
      grad_wpe2.to(TypeToTorch<Element>::type), tol_backward);

  //  if (!isApprox_grad_wte) {
  //    std::ofstream fileOuput("grad_wte1.txt");
  //    fileOuput << grad_wte1 << std::endl;
  //    fileOuput.close();
  //    std::ofstream fileRef("grad_wte2.txt");
  //    fileRef << grad_wte2 << std::endl;
  //    fileRef.close();
  //  }
  //
  //  if (!isApprox_grad_wpe) {
  //    std::ofstream fileOuput("grad_wpe1.txt");
  //    fileOuput << grad_wpe1 << std::endl;
  //    fileOuput.close();
  //    std::ofstream fileRef("grad_wpe2.txt");
  //    fileRef << grad_wpe2 << std::endl;
  //    fileRef.close();
  //  }

  ASSERT_TRUE(isApprox_grad_wpe);
  ASSERT_TRUE(isApprox_grad_wte);

  CHECK_CUDART(cudaFree(DeviceInput));
  CHECK_CUDART(cudaFree(DeviceWte));
  CHECK_CUDART(cudaFree(DeviceWpe));
  CHECK_CUDART(cudaFree(DeviceOutput));
  CHECK_CUDART(cudaFree(Device_grad_output));
  CHECK_CUDART(cudaFree(Device_grad_wte));
  CHECK_CUDART(cudaFree(Device_grad_wpe));
}

TEST_F(TestEmbedding, TestFloat) { TestRoutine<float>(1e-5, 1e-5); }

TEST_F(TestEmbedding, TestHalf) { TestRoutine<half>(1e-5, 1e-2);
}