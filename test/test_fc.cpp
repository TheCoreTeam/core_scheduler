#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "fc.h"
#include "logger.h"

class FcTestFixture : public ::testing::Test {
 protected:
  dllm::Context context;

  void SetUp() override {
    CHECK_CUDART(
        cudaStreamCreateWithFlags(&context.cudaStream, cudaStreamNonBlocking));
    CHECK_CUBLAS(cublasCreate_v2(&context.cublasHandle));
    CHECK_CUBLAS(cublasSetStream_v2(context.cublasHandle, context.cudaStream));
  }

  void TearDown() override {
    CHECK_CUBLAS(cublasDestroy_v2(context.cublasHandle));
    CHECK_CUDART(cudaStreamDestroy(context.cudaStream));
  }
};

namespace {
template <typename ComputeType>
cublasComputeType_t toCublasComputeType() {
  if constexpr (std::is_same_v<ComputeType, double>) {
    return CUBLAS_COMPUTE_64F_PEDANTIC;
  } else if constexpr (std::is_same_v<ComputeType, float>) {
    return CUBLAS_COMPUTE_32F_PEDANTIC;
  }
}
}  // namespace

namespace {
template <typename DataTypeInput, typename DataTypeOutput, typename ComputeType>
void TestForwardT(const dllm::Context &context) {
  const int m = 128, n = 2048, k = 512, s = 3;
  auto shapeX = cute::make_shape(m, s, k);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
  auto shapeW = cute::make_shape(n, k);
  auto layoutW = cute::make_layout(shapeW, cute::GenRowMajor{});
  auto shapeY = cute::make_shape(m, s, n);
  auto layoutY = cute::make_layout(shapeY, cute::GenRowMajor{});

  void *ptrX, *ptrW, *ptrY;
  cudaMalloc(&ptrX, sizeof(DataTypeInput) * cute::size(layoutX));
  cudaMalloc(&ptrW, sizeof(DataTypeInput) * cute::size(layoutW));
  cudaMalloc(&ptrY, sizeof(DataTypeOutput) * cute::size(layoutY));

  dllm::Tensor3D tensorX{ptrX, layoutX, dllm::toDtype<DataTypeInput>(),
                         dllm::CUDA};
  dllm::Tensor2D tensorW{ptrW, layoutW, dllm::toDtype<DataTypeInput>(),
                         dllm::CUDA};
  dllm::Tensor3D tensorY{ptrY, layoutY, dllm::toDtype<DataTypeOutput>(),
                         dllm::CUDA};

  Eigen::Matrix<DataTypeInput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostX(m * s, k), hostW(n, k);
  Eigen::Matrix<DataTypeOutput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostY(m * s, n), refY;

  hostX.setRandom();
  hostW.setRandom();

  refY = (hostX.template cast<ComputeType>() *
          hostW.transpose().template cast<ComputeType>())
             .template cast<DataTypeOutput>();

  cudaMemcpy(ptrX, hostX.data(), sizeof(DataTypeInput) * cute::size(layoutX),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ptrW, hostW.data(), sizeof(DataTypeInput) * cute::size(layoutW),
             cudaMemcpyHostToDevice);

  auto task = dllm::FcNoBias::forward(tensorY, tensorX, tensorW,
                                      toCublasComputeType<ComputeType>());
  task(&context);

  cudaMemcpy(hostY.data(), ptrY, sizeof(DataTypeOutput) * cute::size(layoutY),
             cudaMemcpyDeviceToHost);

  for (int col = 0; col < refY.cols(); ++col) {
    for (int row = 0; row < refY.rows(); ++row) {
      ASSERT_NEAR(hostY(row, col), refY(row, col), 1e-4);
    }
  }

  cudaFree(ptrY);
  cudaFree(ptrW);
  cudaFree(ptrX);
}
}  // namespace

TEST_F(FcTestFixture, TestForwardF32F32F32) {
  TestForwardT<float, float, float>(context);
}
TEST_F(FcTestFixture, TestForwardF64F64F64) {
  TestForwardT<double, double, double>(context);
}

namespace {
template <typename DataTypeInput, typename DataTypeOutput, typename ComputeType>
void TestBackwardWT(const dllm::Context &context) {
  const int m = 128, n = 2048, k = 512, s = 3;
  auto shapeX = cute::make_shape(m, s, k);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
  auto shapeDW = cute::make_shape(n, k);
  auto layoutDW = cute::make_layout(shapeDW, cute::GenRowMajor{});
  auto shapeDY = cute::make_shape(m, s, n);
  auto layoutDY = cute::make_layout(shapeDY, cute::GenRowMajor{});

  void *ptrX, *ptrDW, *ptrDY;
  cudaMalloc(&ptrX, sizeof(DataTypeInput) * cute::size(layoutX));
  cudaMalloc(&ptrDW, sizeof(DataTypeOutput) * cute::size(layoutDW));
  cudaMalloc(&ptrDY, sizeof(DataTypeInput) * cute::size(layoutDY));

  dllm::Tensor3D tensorX{ptrX, layoutX, dllm::toDtype<DataTypeInput>(),
                         dllm::CUDA};
  dllm::Tensor2D tensorDW{ptrDW, layoutDW, dllm::toDtype<DataTypeOutput>(),
                          dllm::CUDA};
  dllm::Tensor3D tensorDY{ptrDY, layoutDY, dllm::toDtype<DataTypeInput>(),
                          dllm::CUDA};

  Eigen::Matrix<DataTypeInput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostX(m * s, k), hostDY(m * s, n);
  Eigen::Matrix<DataTypeOutput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostDW(n, k), refDW;

  hostX.setRandom();
  hostDY.setRandom();

  refDW = (hostDY.transpose().template cast<ComputeType>() *
           hostX.template cast<ComputeType>())
              .template cast<DataTypeOutput>();

  cudaMemcpy(ptrX, hostX.data(), sizeof(DataTypeInput) * cute::size(layoutX),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ptrDY, hostDY.data(), sizeof(DataTypeInput) * cute::size(layoutDY),
             cudaMemcpyHostToDevice);

  auto task = dllm::FcNoBias::backwardW(tensorDW, tensorDY, tensorX,
                                        toCublasComputeType<ComputeType>());
  task(&context);

  cudaMemcpy(hostDW.data(), ptrDW,
             sizeof(DataTypeOutput) * cute::size(layoutDW),
             cudaMemcpyDeviceToHost);

  for (int col = 0; col < refDW.cols(); ++col) {
    for (int row = 0; row < refDW.rows(); ++row) {
      ASSERT_NEAR(hostDW(row, col), refDW(row, col), 1e-4);
    }
  }

  cudaFree(ptrDY);
  cudaFree(ptrDW);
  cudaFree(ptrX);
}
}  // namespace

TEST_F(FcTestFixture, TestBackwardWF32F32F32) {
  TestBackwardWT<float, float, float>(context);
}
TEST_F(FcTestFixture, TestBackwardWF64F64F64) {
  TestBackwardWT<double, double, double>(context);
}

namespace {
template <typename DataTypeInput, typename DataTypeOutput, typename ComputeType>
void TestBackwardXT(const dllm::Context &context) {
  const int m = 128, n = 2048, k = 512, s = 3;
  auto shapeDX = cute::make_shape(m, s, k);
  auto layoutDX = cute::make_layout(shapeDX, cute::GenRowMajor{});
  auto shapeDY = cute::make_shape(m, s, n);
  auto layoutDY = cute::make_layout(shapeDY, cute::GenRowMajor{});
  auto shapeW = cute::make_shape(n, k);
  auto layoutW = cute::make_layout(shapeW, cute::GenRowMajor{});

  void *ptrDX, *ptrDY, *ptrW;
  cudaMalloc(&ptrDX, sizeof(DataTypeOutput) * cute::size(layoutDX));
  cudaMalloc(&ptrDY, sizeof(DataTypeInput) * cute::size(layoutDY));
  cudaMalloc(&ptrW, sizeof(DataTypeInput) * cute::size(layoutW));

  dllm::Tensor3D tensorDX{ptrDX, layoutDX, dllm::toDtype<DataTypeOutput>(),
                          dllm::CUDA};
  dllm::Tensor3D tensorDY{ptrDY, layoutDY, dllm::toDtype<DataTypeInput>(),
                          dllm::CUDA};
  dllm::Tensor2D tensorW{ptrW, layoutW, dllm::toDtype<DataTypeInput>(),
                         dllm::CUDA};

  Eigen::Matrix<DataTypeInput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostDY(m * s, n), hostW(n, k);
  Eigen::Matrix<DataTypeOutput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostDX(m * s, k), refDX;

  hostDY.setRandom();
  hostW.setRandom();

  refDX =
      (hostDY.template cast<ComputeType>() * hostW.template cast<ComputeType>())
          .template cast<DataTypeOutput>();

  cudaMemcpy(ptrDY, hostDY.data(), sizeof(DataTypeInput) * cute::size(layoutDY),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ptrW, hostW.data(), sizeof(DataTypeInput) * cute::size(layoutW),
             cudaMemcpyHostToDevice);

  auto task = dllm::FcNoBias::backwardX(tensorDX, tensorDY, tensorW,
                                        toCublasComputeType<ComputeType>());
  task(&context);

  cudaMemcpy(hostDX.data(), ptrDX,
             sizeof(DataTypeOutput) * cute::size(layoutDX),
             cudaMemcpyDeviceToHost);

  for (int col = 0; col < refDX.cols(); ++col) {
    for (int row = 0; row < refDX.rows(); ++row) {
      ASSERT_NEAR(hostDX(row, col), refDX(row, col), 1e-4);
    }
  }

  cudaFree(ptrDX);
  cudaFree(ptrDY);
  cudaFree(ptrW);
}
}  // namespace

TEST_F(FcTestFixture, TestBackwardXF32F32F32) {
  TestBackwardXT<float, float, float>(context);
}
TEST_F(FcTestFixture, TestBackwardXF64F64F64) {
  TestBackwardXT<double, double, double>(context);
}
