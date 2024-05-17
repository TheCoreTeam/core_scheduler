#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/nn/functional.h>

#include <Eigen/Dense>
#include <fstream>

#include "compute/softmax.h"
#include "logger.h"
#include "tensor.h"

namespace Eigen::internal {
template <>
struct scalar_random_op<nv_half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_random_op)
  inline const nv_half operator()() const {
    return static_cast<nv_half>(random<float>());
  }
};
}  // namespace Eigen::internal

class TestDLLMSoftmax : public ::testing::Test {
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
  void TestRoutine(dllm::TensorIndexType row, dllm::TensorIndexType col);
};

template <typename T>
struct ToTorchType;

template <>
struct ToTorchType<double> {
  constexpr static auto value = torch::kDouble;
};

template <>
struct ToTorchType<float> {
  constexpr static auto value = torch::kFloat;
};

template <>
struct ToTorchType<nv_half> {
  constexpr static auto value = torch::kHalf;
};

template <typename Element>
void TestDLLMSoftmax::TestRoutine(const dllm::TensorIndexType row,
                                  const dllm::TensorIndexType col) {
  using Dtype = std::conditional_t<
      std::is_same_v<Element, nv_half>, torch::Half,
      std::conditional_t<std::is_same_v<Element, nv_bfloat16>, torch::BFloat16,
                         Element>>;
  const double temperature = 1.0;
  torch::manual_seed(1);
  auto dtype = torch::TensorOptions{ToTorchType<Element>::value};
  auto doubleType = torch::TensorOptions{ToTorchType<double>::value};
  auto hostInput = torch::randn({row, col}, dtype);
  auto hostOutput = torch::empty_like(hostInput);
  auto hostRef = torch::softmax(hostInput.to(doubleType), 1);

  auto shape = cute::make_shape(row, col);
  auto layout = cute::make_layout(shape, cute::GenRowMajor{});

  void *DeviceInput, *DeviceOutput;
  CHECK_CUDART(cudaMalloc(&DeviceInput, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaMalloc(&DeviceOutput, sizeof(Element) * cute::size(layout)));
  auto tensorInput = std::make_shared<dllm::Tensor2D>(
      DeviceInput, layout, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorOutput = std::make_shared<dllm::Tensor2D>(
      DeviceOutput, layout, dllm::toDtype<Element>(), dllm::CUDA);

  CHECK_CUDART(cudaMemcpy(DeviceInput, hostInput.data_ptr<Dtype>(),
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyHostToDevice));

  CHECK_CUDART(cudaDeviceSynchronize());

  auto tast =
      dllm::compute::Softmax::forward(tensorInput, tensorOutput, temperature);
  tast(&context);

  CHECK_CUDART(cudaMemcpy(hostOutput.data_ptr<Dtype>(), DeviceOutput,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto isApprox = hostOutput.to(doubleType).allclose(hostRef);

  if (!isApprox) {
    std::ofstream fileOuput("output.txt");
    fileOuput << hostOutput << std::endl;
    fileOuput.close();
    std::ofstream fileRef("ref.txt");
    fileRef << hostRef << std::endl;
    fileRef.close();
  }

  ASSERT_TRUE(isApprox);
  CHECK_CUDART(cudaFree(DeviceInput));
  CHECK_CUDART(cudaFree(DeviceOutput));
}

TEST_F(TestDLLMSoftmax, TestF32_128) { TestRoutine<float>(128, 32); }

TEST_F(TestDLLMSoftmax, TestF32_512) { TestRoutine<float>(512, 32); }

TEST_F(TestDLLMSoftmax, TestF32_1024) { TestRoutine<float>(1024, 32); }

TEST_F(TestDLLMSoftmax, TestF64_128) { TestRoutine<double>(128, 32); }

TEST_F(TestDLLMSoftmax, TestF64_512) { TestRoutine<double>(512, 32); }

TEST_F(TestDLLMSoftmax, TestF64_1024) { TestRoutine<double>(1024, 32); }

TEST_F(TestDLLMSoftmax, TestF16_128) { TestRoutine<nv_half>(128, 32); }

TEST_F(TestDLLMSoftmax, TestF16_512) { TestRoutine<nv_half>(512, 32); }

TEST_F(TestDLLMSoftmax, TestF16_1024) { TestRoutine<nv_half>(1024, 32); }
