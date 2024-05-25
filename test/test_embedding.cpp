#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "compute/embedding.h"
#include "threading/thread_pool_compute.h"

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

class TestEmbedding : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute tp{0, 2};

  template <typename Element>
  void TestRoutine(const double tol_forward, const double tol_backward);
};

template <typename Element>
void TestEmbedding::TestRoutine(const double tol_forward,
                                const double tol_backward) {
  torch::manual_seed(1);
  const int B = 2;
  const int T = 1024;
  const int MaxT = 1024;
  const int vocab = 4095;
  const int d = 512;
  //  const int B = 1;
  //  const int T = 8;
  //  const int MaxT = 8;
  //  const int vocab = 16;
  //  const int d = 8;

  torch::Device device = torch::kCUDA;
  torch::Dtype dtype = TypeToTorch<Element>::type;
  auto wte = torch::randn({vocab, d},
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

  auto input2 = input.detach().clone();
  auto wte2 = wte.detach().clone();

  auto output1 = at::embedding(wte1, input1);

  auto state = dllm::compute::Embedding::init(vocab, d, {}, {});

  auto output2 = std::make_shared<dllm::Tensor>();
  {
    auto task = dllm::compute::Embedding::forward(
        output2, std::make_shared<dllm::Tensor>(input), state);
    tp.submit(std::move(task));
    output2->wait();
  }

  ASSERT_TRUE(torch::allclose(output1, output2->tensor()));
  // // backward check
  // auto grad_output = torch::rand_like(output1);
  //
  // // 计算梯度
  // auto grads = torch::autograd::grad(
  //     {output1}, {wte1, wpe1}, {grad_output}, /*retain_graph=*/false,
  //     /*create_graph=*/false, /*allow_unused=*/true);
  //
  // // Access and print gradients
  // auto grad_wte1 = grads[0];
  // auto grad_wpe1 = grads[1];
  //
  // void *Device_grad_output;
  // void *Device_grad_wte;
  // void *Device_grad_wpe;
  // auto shapeGradOutput = shapeOutput;
  // auto shapeGradWte = shapeWte;
  // auto shapeGradWpe = shapeWpe;
  // auto layoutGradOutput =
  //     cute::make_layout(shapeGradOutput, cute::GenRowMajor{});
  // auto layoutGradWte = cute::make_layout(shapeGradWte, cute::GenRowMajor{});
  // auto layoutGradWpe = cute::make_layout(shapeGradWpe, cute::GenRowMajor{});
  // CHECK_CUDART(cudaMalloc(&Device_grad_output,
  //                         sizeof(Element) * cute::size(layoutGradOutput)));
  // CHECK_CUDART(cudaMalloc(&Device_grad_wte,
  //                         sizeof(Element) * cute::size(layoutGradWte)));
  // CHECK_CUDART(
  //     cudaMalloc(&Device_grad_wpe, sizeof(Element) *
  //     cute::size(shapeGradWpe)));
  // CHECK_CUDART(cudaMemcpy(
  //     Device_grad_output,
  //     grad_output.to(dtype)
  //         .template data_ptr<typename TypeToTorch<Element>::Type>(),
  //     sizeof(Element) * cute::size(layoutGradOutput),
  //     cudaMemcpyHostToDevice));
  // CHECK_CUDART(cudaMemset(Device_grad_wte, 0,
  //                         sizeof(Element) * cute::size(layoutGradWte)));
  // CHECK_CUDART(cudaMemset(Device_grad_wpe, 0,
  //                         sizeof(Element) * cute::size(layoutGradWpe)));
  // auto tensorGradOutput = std::make_shared<dllm::Tensor3D>(
  //     Device_grad_output, layoutOutput, dllm::toDtype<Element>(),
  //     dllm::CUDA);
  // auto tensorGradWte = std::make_shared<dllm::Tensor2D>(
  //     Device_grad_wte, layoutGradWte, dllm::toDtype<Element>(), dllm::CUDA);
  // auto tensorGradWpe = std::make_shared<dllm::Tensor2D>(
  //     Device_grad_wpe, layoutGradWpe, dllm::toDtype<Element>(), dllm::CUDA);
  //
  // CHECK_CUDART(cudaDeviceSynchronize());
  //
  // auto task_backward = dllm::compute::embedding::backward(
  //     tensorInput, tensorGradWte, tensorGradWpe, tensorGradOutput);
  // task_backward(&context);
  //
  // auto grad_wte2 =
  //     torch::empty_like(grad_wte1, torch::TensorOptions().dtype(dtype));
  // auto grad_wpe2 =
  //     torch::empty_like(grad_wpe1, torch::TensorOptions().dtype(dtype));
  //
  // CHECK_CUDART(cudaMemcpy(
  //     grad_wte2.template data_ptr<typename TypeToTorch<Element>::Type>(),
  //     Device_grad_wte, sizeof(Element) * cute::size(layoutGradWte),
  //     cudaMemcpyDeviceToHost));
  // CHECK_CUDART(cudaMemcpy(
  //     grad_wpe2.template data_ptr<typename TypeToTorch<Element>::Type>(),
  //     Device_grad_wpe, sizeof(Element) * cute::size(layoutGradWpe),
  //     cudaMemcpyDeviceToHost));
  //
  // CHECK_CUDART(cudaDeviceSynchronize());
  //
  // auto isApprox_grad_wte = grad_wte1.allclose(
  //     grad_wte2.to(TypeToTorch<Element>::type), tol_backward);
  // auto isApprox_grad_wpe = grad_wpe1.allclose(
  //     grad_wpe2.to(TypeToTorch<Element>::type), tol_backward);
}

TEST_F(TestEmbedding, TestFloat) { TestRoutine<float>(1e-5, 1e-5); }

TEST_F(TestEmbedding, TestHalf) { TestRoutine<half>(1e-5, 1e-2);
}