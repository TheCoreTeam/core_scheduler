//#include <cublas_v2.h>
//#include <cuda_runtime.h>
//#include <gtest/gtest.h>
//#include <math_constants.h>
//
//#include <Eigen/Dense>
//#include <cmath>
//
//#include "compute/gelu.h"
//#include "logger.h"
//#include "threading/thread_pool_compute.h"
//#include "util.h"
//
//namespace Eigen::internal {
//template <>
//struct scalar_random_op<nv_half> {
//  EIGEN_EMPTY_STRUCT_CTOR(scalar_random_op)
//  inline const nv_half operator()() const {
//    return static_cast<nv_half>(random<float>());
//  }
//};
//}  // namespace Eigen::internal
//
//class GeLUTestFixture : public ::testing::Test {
// protected:
//  dllm::ContextCompute context{};
//
//  void SetUp() override {
//    CHECK_CUDART(
//        cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
//    CHECK_CUBLAS(cublasCreate_v2(&context.cublasHandle));
//    CHECK_CUBLAS(cublasSetStream_v2(context.cublasHandle, context.cudaStream));
//  }
//
//  void TearDown() override {
//    CHECK_CUBLAS(cublasDestroy_v2(context.cublasHandle));
//    CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
//  }
//};
//
//namespace {
//template <typename DataType>
//void TestForwardT(const dllm::ContextCompute &context) {
//  const int m = 1, k = 1, s = 3;
//  auto shapeX = cute::make_shape(m, s, k);
//  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
//  auto shapeY = cute::make_shape(m, s, k);
//  auto layoutY = cute::make_layout(shapeY, cute::GenRowMajor{});
//
//  void *ptrX, *ptrY;
//  CHECK_CUDART(cudaMalloc(&ptrX, sizeof(DataType) * cute::size(layoutX)));
//  CHECK_CUDART(cudaMalloc(&ptrY, sizeof(DataType) * cute::size(layoutY)));
//
//  auto tensorX = std::make_shared<dllm::Tensor3D>(
//      ptrX, layoutX, dllm::toDtype<DataType>(), dllm::CUDA);
//  auto tensorY = std::make_shared<dllm::Tensor3D>(
//      ptrY, layoutY, dllm::toDtype<DataType>(), dllm::CUDA);
//
//  Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//      hostX(m * s, k);
//  Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//      hostY(m * s, k), refY;
//
//  hostX.setRandom();
//
//  DataType alpha = static_cast<DataType>(0.044715);
//  DataType half_one = static_cast<DataType>(0.5);
//  DataType one = static_cast<DataType>(1);
//  DataType sqrt_term = static_cast<DataType>(CUDART_SQRT_2OPI);
//
////  auto tanh_term = sqrt_term * (hostX.array().template cast<DataType>() + alpha * hostX.array().template cast<DataType>().pow(3)).template cast<DataType>().array().template cast<DataType>().tanh();
////  refY = half_one * hostX.array().template cast<DataType>() * (one + tanh_term.template cast<DataType>().array()).template cast<DataType>();
//  refY = (half_one * hostX.array() * (one + (sqrt_term * hostX.array() + alpha * hostX.array().cube()).array().tanh())).matrix();
//
//
//  CHECK_CUDART(cudaMemcpy(ptrX, hostX.data(),
//                          sizeof(DataType) * cute::size(layoutX),
//                          cudaMemcpyHostToDevice));
//  CHECK_CUDART(cudaDeviceSynchronize());
//
//  auto tensorX1D = dllm::util::flatten<1>(tensorX);
//  auto tensorY1D = dllm::util::flatten<1>(tensorY);
//
//  auto task = dllm::compute::GeLU::GeLU(
//      tensorY1D, tensorX1D);
//  tensorX.reset();
//  tensorY.reset();
//  task(&context);
//
//  CHECK_CUDART(cudaMemcpy(hostY.data(), ptrY,
//                          sizeof(DataType) * cute::size(layoutY),
//                          cudaMemcpyDeviceToHost));
//  CHECK_CUDART(cudaDeviceSynchronize());
//
//  for (int col = 0; col < refY.cols(); ++col) {
//    for (int row = 0; row < refY.rows(); ++row) {
//      std::cout<<hostY(row,col)<<std::endl;
//      std::cout<<refY(row,col)<<std::endl;
//      ASSERT_NEAR(hostY(row, col), refY(row, col), 1);
//    }
//  }
//
//  CHECK_CUDART(cudaFree(ptrY));
//  CHECK_CUDART(cudaFree(ptrX));
//}
//}  // namespace
////
////TEST_F(GeLUTestFixture, TestForwardF16) {
////  TestForwardT<nv_half>(context);
////}
//TEST_F(GeLUTestFixture, TestForwardF32) {
//  TestForwardT<float>(context);
//}
//TEST_F(GeLUTestFixture, TestForwardF64) {
//  TestForwardT<double>(context);
//}
//
//class GeLUThreadPoolComputeTestFixture : public ::testing::Test {
// protected:
//  dllm::ThreadPoolCompute threadPool{0, 1};
//};
//
//namespace {
//template <typename DataType>
//void TestThreadPoolComputeForwardT(dllm::ThreadPoolCompute &threadPool) {
//  const int m = 128, k = 512, s = 3;
//  auto shapeX = cute::make_shape(m, s, k);
//  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
//  auto shapeY = cute::make_shape(m, s, k);
//  auto layoutY = cute::make_layout(shapeY, cute::GenRowMajor{});
//
//  void *ptrX, *ptrY;
//  CHECK_CUDART(cudaMalloc(&ptrX, sizeof(DataType) * cute::size(layoutX)));
//  CHECK_CUDART(cudaMalloc(&ptrY, sizeof(DataType) * cute::size(layoutY)));
//
//  auto tensorX = std::make_shared<dllm::Tensor3D>(
//      ptrX, layoutX, dllm::toDtype<DataType>(), dllm::CUDA);
//  auto tensorY = std::make_shared<dllm::Tensor3D>(
//      ptrY, layoutY, dllm::toDtype<DataType>(), dllm::CUDA);
//
//  Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//      hostX(m * s, k);
//  Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
//      hostY(m * s, k), refY;
//
//  hostX.setRandom();
//
//  DataType alpha = 0.044715;
//  DataType half_one = 0.5;
//  DataType one = 1;
//  DataType sqrt_term = CUDART_SQRT_2OPI;
//
//  auto tanh_term = sqrt_term * (hostX.array().template cast<DataType>() + alpha * hostX.array().template cast<DataType>().pow(3)).template cast<DataType>().array().template cast<DataType>().tanh();
//  refY = half_one * hostX.array().template cast<DataType>() * (one + tanh_term.template cast<DataType>().array()).template cast<DataType>();
//
//  CHECK_CUDART(cudaMemcpy(ptrX, hostX.data(),
//                          sizeof(DataType) * cute::size(layoutX),
//                          cudaMemcpyHostToDevice));
//  CHECK_CUDART(cudaDeviceSynchronize());
//
//  auto tensorY2D = dllm::util::flatten<1>(tensorY);
//  auto tensorX2D = dllm::util::flatten<1>(tensorX);
//
//  auto task = dllm::compute::GeLU::GeLU(
//      tensorY2D, tensorX2D);
//  tensorX.reset();
//  auto future = threadPool.submit(std::move(task));
//  future->wait();
//
//  CHECK_CUDART(cudaMemcpy(hostY.data(), ptrY,
//                          sizeof(DataType) * cute::size(layoutY),
//                          cudaMemcpyDeviceToHost));
//  CHECK_CUDART(cudaDeviceSynchronize());
//
//  for (int col = 0; col < refY.cols(); ++col) {
//    for (int row = 0; row < refY.rows(); ++row) {
//      ASSERT_NEAR(hostY(row, col), refY(row, col), 1e-4);
//    }
//  }
//
//  CHECK_CUDART(cudaFree(ptrY));
//  CHECK_CUDART(cudaFree(ptrX));
//}
//}  // namespace
//
////TEST_F(GeLUThreadPoolComputeTestFixture, TestForwardF16) {
////  TestThreadPoolComputeForwardT<nv_half>(threadPool);
////}
//TEST_F(GeLUThreadPoolComputeTestFixture, TestForwardF32) {
//  TestThreadPoolComputeForwardT<float>(threadPool);
//}
//TEST_F(GeLUThreadPoolComputeTestFixture, TestForwardF64) {
//  TestThreadPoolComputeForwardT<double>(threadPool);
//}

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <fstream>
#include "compute/gelu.h"
#include "tensor.h"
#include "logger.h"
#include <math_constants.h>

namespace Eigen::internal {
template <>
struct scalar_random_op<nv_half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_random_op);
  inline const nv_half operator()() const {
    return static_cast<nv_half>(random<float>());
  }
};
}  // namespace Eigen::internal

class TestDLLMRelu : public ::testing::Test {
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
void TestDLLMRelu::TestRoutine(const int size) {
  using DataType=Element;
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

  CHECK_CUDART(cudaMemcpy(DeviceInput, hostInput.data(), sizeof(Element) * cute::size(layout),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemset(DeviceOutput, 0, sizeof(Element) * cute::size(layout)));

  CHECK_CUDART(cudaDeviceSynchronize());


    DataType alpha = static_cast<DataType>(0.044715);
    DataType half_one = static_cast<DataType>(0.5);
    DataType one = static_cast<DataType>(1);
    DataType sqrt_term = static_cast<DataType>(CUDART_SQRT_2OPI);

    constexpr auto inv_sqrt_2 = 0.7071067811865475;
    hostRef = float{0.5} * hostInput * (float{1} + (hostInput.template cast<float>().array() * float{inv_sqrt_2}).erf()).matrix();

  auto tast = dllm::compute::GeLU(tensorOutput,tensorInput);
  tast(&context);

  CHECK_CUDART(cudaMemcpy(hostOutput.data(), DeviceOutput, sizeof(Element) * cute::size(layout),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto isApprox = hostOutput.isApprox(hostRef);

  if(!isApprox) {
    if constexpr (std::is_same_v<Element, nv_half>) {
      std::ofstream fileOuput("output.txt");
      fileOuput << hostOutput.template cast<float>() << std::endl;
      fileOuput.close();
      std::ofstream fileRef("ref.txt");
      fileRef << hostRef.template cast<float>() << std::endl;
      fileRef.close();
    } else {
      std::ofstream fileOuput("output.txt");
      fileOuput << hostOutput << std::endl;
      fileOuput.close();
      std::ofstream fileRef("ref.txt");
      fileRef << hostRef << std::endl;
      fileRef.close();
    }
  }

  ASSERT_TRUE(isApprox);
  CHECK_CUDART(cudaFree(DeviceInput));
  CHECK_CUDART(cudaFree(DeviceOutput));
}

TEST_F(TestDLLMRelu, TestF32_128) {
  TestRoutine<float>(128);
}

TEST_F(TestDLLMRelu, TestF32_512) {
  TestRoutine<float>(512);
}

TEST_F(TestDLLMRelu, TestF32_1024) {
  TestRoutine<float>(1024);
}

TEST_F(TestDLLMRelu, TestF64_128) {
  TestRoutine<double>(128);
}

TEST_F(TestDLLMRelu, TestF64_512) {
  TestRoutine<double>(512);
}

TEST_F(TestDLLMRelu, TestF64_1024) {
  TestRoutine<double>(1024);
}

//TEST_F(TestDLLMRelu, TestF16_128) {
//  TestRoutine<nv_half>(128);
//}

//TEST_F(TestDLLMRelu, TestF16_512) {
//  TestRoutine<nv_half>(512);
//}
//
//TEST_F(TestDLLMRelu, TestF16_1024) {
//  TestRoutine<nv_half>(1024);
//}


