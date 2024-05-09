#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <math_constants.h>

#include <Eigen/Dense>
#include <cmath>

#include "compute/gelu.h"
#include "logger.h"
#include "tensor.h"

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
  void TestRoutine(const int size);
};

template <typename Element>
void TestDLLMGelu::TestRoutine(const int size) {
  using DataType = Element;
  Eigen::Vector<Element, Eigen::Dynamic> hostInput(size);
  Eigen::Vector<Element, Eigen::Dynamic> hostOutput(size);
  Eigen::Vector<Element, Eigen::Dynamic> hostRef(size);
  hostInput.setRandom();
  hostOutput.setZero();

  auto shape = cute::make_shape(size);
  auto layout = cute::make_layout(shape, cute::GenRowMajor{});

  void *DeviceInput, *DeviceOutput;
  CHECK_CUDART(cudaMalloc(&DeviceInput, sizeof(Element) * cute::size(layout)));
  CHECK_CUDART(cudaMalloc(&DeviceOutput, sizeof(Element) * cute::size(layout)));
  auto tensorInput = std::make_shared<dllm::Tensor1D>(
      DeviceInput, layout, dllm::toDtype<Element>(), dllm::CUDA);
  auto tensorOutput = std::make_shared<dllm::Tensor1D>(
      DeviceOutput, layout, dllm::toDtype<Element>(), dllm::CUDA);

  CHECK_CUDART(cudaMemcpy(DeviceInput, hostInput.data(),
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(
      cudaMemset(DeviceOutput, 0, sizeof(Element) * cute::size(layout)));

  CHECK_CUDART(cudaDeviceSynchronize());

  constexpr auto useDouble = sizeof(Element) > sizeof(float);
  using TargetType = std::conditional_t<useDouble, double, float>;
  constexpr TargetType inv_sqrt_2 = 1. / std::sqrt(2.);
  hostRef = hostInput.unaryExpr([=](Element x) -> Element {
    return static_cast<TargetType>(x) * static_cast<TargetType>(0.5) *
           (static_cast<TargetType>(1.) +
            erf(static_cast<TargetType>(x) * inv_sqrt_2));
  });

  auto tast = dllm::compute::GeLU(tensorOutput, tensorInput);
  tast(&context);

  CHECK_CUDART(cudaMemcpy(hostOutput.data(), DeviceOutput,
                          sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto isApprox = hostOutput.isApprox(hostRef);

  ASSERT_TRUE(isApprox);
  CHECK_CUDART(cudaFree(DeviceInput));
  CHECK_CUDART(cudaFree(DeviceOutput));
}

TEST_F(TestDLLMGelu, TestF32_128) { TestRoutine<float>(128); }

TEST_F(TestDLLMGelu, TestF32_512) { TestRoutine<float>(512); }

TEST_F(TestDLLMGelu, TestF32_1024) { TestRoutine<float>(1024); }

TEST_F(TestDLLMGelu, TestF64_128) { TestRoutine<double>(128); }

TEST_F(TestDLLMGelu, TestF64_512) { TestRoutine<double>(512); }

TEST_F(TestDLLMGelu, TestF64_1024) { TestRoutine<double>(1024); }

TEST_F(TestDLLMGelu, TestF16_128) { TestRoutine<nv_half>(128); }

TEST_F(TestDLLMGelu, TestF16_512) { TestRoutine<nv_half>(512); }

TEST_F(TestDLLMGelu, TestF16_1024) { TestRoutine<nv_half>(1024); }
