#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "compute/fc.h"
#include "logger.h"
#include "threading/thread_pool_compute.h"
#include "util.h"

namespace Eigen::internal {
template <>
struct scalar_random_op<nv_half> {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_random_op)
  inline const nv_half operator()() const {
    return static_cast<nv_half>(random<float>());
  }
};
}  // namespace Eigen::internal

class FcTestFixture : public ::testing::Test {
 protected:
  dllm::ContextCompute context{};

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
void TestForwardT(const dllm::ContextCompute &context) {
  const dllm::TensorIndexType m = 128, n = 2048, k = 512, s = 3;
  auto shapeX = cute::make_shape(m, s, k);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
  auto shapeW = cute::make_shape(n, k);
  auto layoutW = cute::make_layout(shapeW, cute::GenRowMajor{});
  auto shapeY = cute::make_shape(m, s, n);
  auto layoutY = cute::make_layout(shapeY, cute::GenRowMajor{});

  void *ptrX, *ptrW, *ptrY;
  CHECK_CUDART(cudaMalloc(&ptrX, sizeof(DataTypeInput) * cute::size(layoutX)));
  CHECK_CUDART(cudaMalloc(&ptrW, sizeof(DataTypeInput) * cute::size(layoutW)));
  CHECK_CUDART(cudaMalloc(&ptrY, sizeof(DataTypeOutput) * cute::size(layoutY)));

  auto tensorX = std::make_shared<dllm::Tensor3D>(
      ptrX, layoutX, dllm::toDtype<DataTypeInput>(), dllm::CUDA);
  auto tensorW = std::make_shared<dllm::Tensor2D>(
      ptrW, layoutW, dllm::toDtype<DataTypeInput>(), dllm::CUDA);
  auto tensorY = std::make_shared<dllm::Tensor3D>(
      ptrY, layoutY, dllm::toDtype<DataTypeOutput>(), dllm::CUDA);

  Eigen::Matrix<DataTypeInput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostX(m * s, k), hostW(n, k);
  Eigen::Matrix<DataTypeOutput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostY(m * s, n), refY;

  hostX.setRandom();
  hostW.setRandom();

  refY = (hostX.template cast<ComputeType>() *
          hostW.transpose().template cast<ComputeType>())
             .template cast<DataTypeOutput>();

  CHECK_CUDART(cudaMemcpy(ptrX, hostX.data(),
                          sizeof(DataTypeInput) * cute::size(layoutX),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(ptrW, hostW.data(),
                          sizeof(DataTypeInput) * cute::size(layoutW),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorY2D = dllm::util::flatten<2>(tensorY);
  auto tensorX2D = dllm::util::flatten<2>(tensorX);

  auto task = dllm::compute::FcNoBias::forward(
      tensorY2D, tensorX2D, tensorW, toCublasComputeType<ComputeType>());
  tensorY.reset();
  tensorX.reset();
  tensorW.reset();
  task(&context);

  CHECK_CUDART(cudaMemcpy(hostY.data(), ptrY,
                          sizeof(DataTypeOutput) * cute::size(layoutY),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  for (int col = 0; col < refY.cols(); ++col) {
    for (int row = 0; row < refY.rows(); ++row) {
      ASSERT_NEAR(hostY(row, col), refY(row, col), 1e-4);
    }
  }

  CHECK_CUDART(cudaFree(ptrY));
  CHECK_CUDART(cudaFree(ptrW));
  CHECK_CUDART(cudaFree(ptrX));
}
}  // namespace

TEST_F(FcTestFixture, TestForwardF16F32F32) {
  TestForwardT<nv_half, float, float>(context);
}
TEST_F(FcTestFixture, TestForwardF32F32F32) {
  TestForwardT<float, float, float>(context);
}
TEST_F(FcTestFixture, TestForwardF64F64F64) {
  TestForwardT<double, double, double>(context);
}

namespace {
template <typename DataTypeInput, typename DataTypeOutput, typename ComputeType>
void TestBackwardWT(const dllm::ContextCompute &context) {
  const dllm::TensorIndexType m = 128, n = 2048, k = 512, s = 3;
  auto shapeX = cute::make_shape(m, s, k);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
  auto shapeDW = cute::make_shape(n, k);
  auto layoutDW = cute::make_layout(shapeDW, cute::GenRowMajor{});
  auto shapeDY = cute::make_shape(m, s, n);
  auto layoutDY = cute::make_layout(shapeDY, cute::GenRowMajor{});

  void *ptrX, *ptrDW, *ptrDY;
  CHECK_CUDART(cudaMalloc(&ptrX, sizeof(DataTypeInput) * cute::size(layoutX)));
  CHECK_CUDART(
      cudaMalloc(&ptrDW, sizeof(DataTypeOutput) * cute::size(layoutDW)));
  CHECK_CUDART(
      cudaMalloc(&ptrDY, sizeof(DataTypeInput) * cute::size(layoutDY)));

  auto tensorX = std::make_shared<dllm::Tensor3D>(
      ptrX, layoutX, dllm::toDtype<DataTypeInput>(), dllm::CUDA);
  auto tensorDW = std::make_shared<dllm::Tensor2D>(
      ptrDW, layoutDW, dllm::toDtype<DataTypeOutput>(), dllm::CUDA);
  auto tensorDY = std::make_shared<dllm::Tensor3D>(
      ptrDY, layoutDY, dllm::toDtype<DataTypeInput>(), dllm::CUDA);

  Eigen::Matrix<DataTypeInput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostX(m * s, k), hostDY(m * s, n);
  Eigen::Matrix<DataTypeOutput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostDW(n, k), refDW;

  hostX.setRandom();
  hostDY.setRandom();

  refDW = (hostDY.transpose().template cast<ComputeType>() *
           hostX.template cast<ComputeType>())
              .template cast<DataTypeOutput>();

  CHECK_CUDART(cudaMemcpy(ptrX, hostX.data(),
                          sizeof(DataTypeInput) * cute::size(layoutX),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(ptrDY, hostDY.data(),
                          sizeof(DataTypeInput) * cute::size(layoutDY),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorDY2D = dllm::util::flatten<2>(tensorDY);
  auto tensorX2D = dllm::util::flatten<2>(tensorX);
  auto task = dllm::compute::FcNoBias::backwardW(
      tensorDW, tensorDY2D, tensorX2D, toCublasComputeType<ComputeType>());
  tensorDW.reset();
  tensorDY.reset();
  tensorX.reset();
  task(&context);

  CHECK_CUDART(cudaMemcpy(hostDW.data(), ptrDW,
                          sizeof(DataTypeOutput) * cute::size(layoutDW),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  for (int col = 0; col < refDW.cols(); ++col) {
    for (int row = 0; row < refDW.rows(); ++row) {
      ASSERT_NEAR(hostDW(row, col), refDW(row, col), 1e-4);
    }
  }

  CHECK_CUDART(cudaFree(ptrDY));
  CHECK_CUDART(cudaFree(ptrDW));
  CHECK_CUDART(cudaFree(ptrX));
}
}  // namespace

TEST_F(FcTestFixture, TestBackwardWF16F32F32) {
  TestBackwardWT<nv_half, float, float>(context);
}
TEST_F(FcTestFixture, TestBackwardWF32F32F32) {
  TestBackwardWT<float, float, float>(context);
}
TEST_F(FcTestFixture, TestBackwardWF64F64F64) {
  TestBackwardWT<double, double, double>(context);
}

namespace {
template <typename DataTypeInput, typename DataTypeOutput, typename ComputeType>
void TestBackwardXT(const dllm::ContextCompute &context) {
  const dllm::TensorIndexType m = 128, n = 2048, k = 512, s = 3;
  auto shapeDX = cute::make_shape(m, s, k);
  auto layoutDX = cute::make_layout(shapeDX, cute::GenRowMajor{});
  auto shapeDY = cute::make_shape(m, s, n);
  auto layoutDY = cute::make_layout(shapeDY, cute::GenRowMajor{});
  auto shapeW = cute::make_shape(n, k);
  auto layoutW = cute::make_layout(shapeW, cute::GenRowMajor{});

  void *ptrDX, *ptrDY, *ptrW;
  CHECK_CUDART(
      cudaMalloc(&ptrDX, sizeof(DataTypeOutput) * cute::size(layoutDX)));
  CHECK_CUDART(
      cudaMalloc(&ptrDY, sizeof(DataTypeInput) * cute::size(layoutDY)));
  CHECK_CUDART(cudaMalloc(&ptrW, sizeof(DataTypeInput) * cute::size(layoutW)));

  auto tensorDX = std::make_shared<dllm::Tensor3D>(
      ptrDX, layoutDX, dllm::toDtype<DataTypeOutput>(), dllm::CUDA);
  auto tensorDY = std::make_shared<dllm::Tensor3D>(
      ptrDY, layoutDY, dllm::toDtype<DataTypeInput>(), dllm::CUDA);
  auto tensorW = std::make_shared<dllm::Tensor2D>(
      ptrW, layoutW, dllm::toDtype<DataTypeInput>(), dllm::CUDA);

  Eigen::Matrix<DataTypeInput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostDY(m * s, n), hostW(n, k);
  Eigen::Matrix<DataTypeOutput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostDX(m * s, k), refDX;

  hostDY.setRandom();
  hostW.setRandom();

  refDX =
      (hostDY.template cast<ComputeType>() * hostW.template cast<ComputeType>())
          .template cast<DataTypeOutput>();

  CHECK_CUDART(cudaMemcpy(ptrDY, hostDY.data(),
                          sizeof(DataTypeInput) * cute::size(layoutDY),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(ptrW, hostW.data(),
                          sizeof(DataTypeInput) * cute::size(layoutW),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorDX2D = dllm::util::flatten<2>(tensorDX);
  auto tensorDY2D = dllm::util::flatten<2>(tensorDY);
  auto task = dllm::compute::FcNoBias::backwardX(
      tensorDX2D, tensorDY2D, tensorW, toCublasComputeType<ComputeType>());
  tensorDX.reset();
  tensorDY.reset();
  tensorW.reset();
  task(&context);

  CHECK_CUDART(cudaMemcpy(hostDX.data(), ptrDX,
                          sizeof(DataTypeOutput) * cute::size(layoutDX),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  for (int col = 0; col < refDX.cols(); ++col) {
    for (int row = 0; row < refDX.rows(); ++row) {
      ASSERT_NEAR(hostDX(row, col), refDX(row, col), 1e-4);
    }
  }

  CHECK_CUDART(cudaFree(ptrDX));
  CHECK_CUDART(cudaFree(ptrDY));
  CHECK_CUDART(cudaFree(ptrW));
}
}  // namespace

TEST_F(FcTestFixture, TestBackwardXF16F32F32) {
  TestBackwardXT<nv_half, float, float>(context);
}
TEST_F(FcTestFixture, TestBackwardXF32F32F32) {
  TestBackwardXT<float, float, float>(context);
}
TEST_F(FcTestFixture, TestBackwardXF64F64F64) {
  TestBackwardXT<double, double, double>(context);
}

class FcThreadPoolComputeTestFixture : public ::testing::Test {
 protected:
  dllm::ThreadPoolCompute threadPool{0, 1};
};

namespace {
template <typename DataTypeInput, typename DataTypeOutput, typename ComputeType>
void TestThreadPoolComputeForwardT(dllm::ThreadPoolCompute &threadPool) {
  const dllm::TensorIndexType m = 128, n = 2048, k = 512, s = 3;
  auto shapeX = cute::make_shape(m, s, k);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
  auto shapeW = cute::make_shape(n, k);
  auto layoutW = cute::make_layout(shapeW, cute::GenRowMajor{});
  auto shapeY = cute::make_shape(m, s, n);
  auto layoutY = cute::make_layout(shapeY, cute::GenRowMajor{});

  void *ptrX, *ptrW, *ptrY;
  CHECK_CUDART(cudaMalloc(&ptrX, sizeof(DataTypeInput) * cute::size(layoutX)));
  CHECK_CUDART(cudaMalloc(&ptrW, sizeof(DataTypeInput) * cute::size(layoutW)));
  CHECK_CUDART(cudaMalloc(&ptrY, sizeof(DataTypeOutput) * cute::size(layoutY)));

  auto tensorX = std::make_shared<dllm::Tensor3D>(
      ptrX, layoutX, dllm::toDtype<DataTypeInput>(), dllm::CUDA);
  auto tensorW = std::make_shared<dllm::Tensor2D>(
      ptrW, layoutW, dllm::toDtype<DataTypeInput>(), dllm::CUDA);
  auto tensorY = std::make_shared<dllm::Tensor3D>(
      ptrY, layoutY, dllm::toDtype<DataTypeOutput>(), dllm::CUDA);

  Eigen::Matrix<DataTypeInput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostX(m * s, k), hostW(n, k);
  Eigen::Matrix<DataTypeOutput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostY(m * s, n), refY;

  hostX.setRandom();
  hostW.setRandom();

  refY = (hostX.template cast<ComputeType>() *
          hostW.transpose().template cast<ComputeType>())
             .template cast<DataTypeOutput>();

  CHECK_CUDART(cudaMemcpy(ptrX, hostX.data(),
                          sizeof(DataTypeInput) * cute::size(layoutX),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(ptrW, hostW.data(),
                          sizeof(DataTypeInput) * cute::size(layoutW),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorY2D = dllm::util::flatten<2>(tensorY);
  auto tensorX2D = dllm::util::flatten<2>(tensorX);
  auto task = dllm::compute::FcNoBias::forward(
      tensorY2D, tensorX2D, tensorW, toCublasComputeType<ComputeType>());
  auto future = tensorY->future;
  tensorY.reset();
  tensorX.reset();
  tensorW.reset();
  threadPool.submit(std::move(task));
  {
    dllm::util::FutureGuard{future->rFuture};
    dllm::util::FutureGuard{future->wFuture};
  }

  CHECK_CUDART(cudaMemcpy(hostY.data(), ptrY,
                          sizeof(DataTypeOutput) * cute::size(layoutY),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  for (int col = 0; col < refY.cols(); ++col) {
    for (int row = 0; row < refY.rows(); ++row) {
      ASSERT_NEAR(hostY(row, col), refY(row, col), 1e-4);
    }
  }

  CHECK_CUDART(cudaFree(ptrY));
  CHECK_CUDART(cudaFree(ptrW));
  CHECK_CUDART(cudaFree(ptrX));
}
}  // namespace

TEST_F(FcThreadPoolComputeTestFixture, TestForwardF16F32F32) {
  TestThreadPoolComputeForwardT<nv_half, float, float>(threadPool);
}
TEST_F(FcThreadPoolComputeTestFixture, TestForwardF32F32F32) {
  TestThreadPoolComputeForwardT<float, float, float>(threadPool);
}
TEST_F(FcThreadPoolComputeTestFixture, TestForwardF64F64F64) {
  TestThreadPoolComputeForwardT<double, double, double>(threadPool);
}

namespace {
template <typename DataTypeInput, typename DataTypeOutput, typename ComputeType>
void TestThreadPoolComputeBackwardWT(dllm::ThreadPoolCompute &threadPool) {
  const dllm::TensorIndexType m = 128, n = 2048, k = 512, s = 3;
  auto shapeX = cute::make_shape(m, s, k);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
  auto shapeDW = cute::make_shape(n, k);
  auto layoutDW = cute::make_layout(shapeDW, cute::GenRowMajor{});
  auto shapeDY = cute::make_shape(m, s, n);
  auto layoutDY = cute::make_layout(shapeDY, cute::GenRowMajor{});

  void *ptrX, *ptrDW, *ptrDY;
  CHECK_CUDART(cudaMalloc(&ptrX, sizeof(DataTypeInput) * cute::size(layoutX)));
  CHECK_CUDART(
      cudaMalloc(&ptrDW, sizeof(DataTypeOutput) * cute::size(layoutDW)));
  CHECK_CUDART(
      cudaMalloc(&ptrDY, sizeof(DataTypeInput) * cute::size(layoutDY)));

  auto tensorX = std::make_shared<dllm::Tensor3D>(
      ptrX, layoutX, dllm::toDtype<DataTypeInput>(), dllm::CUDA);
  auto tensorDW = std::make_shared<dllm::Tensor2D>(
      ptrDW, layoutDW, dllm::toDtype<DataTypeOutput>(), dllm::CUDA);
  auto tensorDY = std::make_shared<dllm::Tensor3D>(
      ptrDY, layoutDY, dllm::toDtype<DataTypeInput>(), dllm::CUDA);

  Eigen::Matrix<DataTypeInput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostX(m * s, k), hostDY(m * s, n);
  Eigen::Matrix<DataTypeOutput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostDW(n, k), refDW;

  hostX.setRandom();
  hostDY.setRandom();

  refDW = (hostDY.transpose().template cast<ComputeType>() *
           hostX.template cast<ComputeType>())
              .template cast<DataTypeOutput>();

  CHECK_CUDART(cudaMemcpy(ptrX, hostX.data(),
                          sizeof(DataTypeInput) * cute::size(layoutX),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(ptrDY, hostDY.data(),
                          sizeof(DataTypeInput) * cute::size(layoutDY),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorDY2D = dllm::util::flatten<2>(tensorDY);
  auto tensorX2D = dllm::util::flatten<2>(tensorX);
  auto task = dllm::compute::FcNoBias::backwardW(
      tensorDW, tensorDY2D, tensorX2D, toCublasComputeType<ComputeType>());
  auto future = tensorX->future;
  tensorDW.reset();
  tensorDY.reset();
  tensorX.reset();
  threadPool.submit(std::move(task));
  {
    dllm::util::FutureGuard{future->rFuture};
    dllm::util::FutureGuard{future->wFuture};
  }

  CHECK_CUDART(cudaMemcpy(hostDW.data(), ptrDW,
                          sizeof(DataTypeOutput) * cute::size(layoutDW),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  for (int col = 0; col < refDW.cols(); ++col) {
    for (int row = 0; row < refDW.rows(); ++row) {
      ASSERT_NEAR(hostDW(row, col), refDW(row, col), 1e-4);
    }
  }

  CHECK_CUDART(cudaFree(ptrDY));
  CHECK_CUDART(cudaFree(ptrDW));
  CHECK_CUDART(cudaFree(ptrX));
}
}  // namespace

TEST_F(FcThreadPoolComputeTestFixture, TestBackwardWF16F32F32) {
  TestThreadPoolComputeBackwardWT<nv_half, float, float>(threadPool);
}
TEST_F(FcThreadPoolComputeTestFixture, TestBackwardWF32F32F32) {
  TestThreadPoolComputeBackwardWT<float, float, float>(threadPool);
}
TEST_F(FcThreadPoolComputeTestFixture, TestBackwardWF64F64F64) {
  TestThreadPoolComputeBackwardWT<double, double, double>(threadPool);
}

namespace {
template <typename DataTypeInput, typename DataTypeOutput, typename ComputeType>
void TestThreadPoolComputeBackwardXT(dllm::ThreadPoolCompute &threadPool) {
  const dllm::TensorIndexType m = 128, n = 2048, k = 512, s = 3;
  auto shapeDX = cute::make_shape(m, s, k);
  auto layoutDX = cute::make_layout(shapeDX, cute::GenRowMajor{});
  auto shapeDY = cute::make_shape(m, s, n);
  auto layoutDY = cute::make_layout(shapeDY, cute::GenRowMajor{});
  auto shapeW = cute::make_shape(n, k);
  auto layoutW = cute::make_layout(shapeW, cute::GenRowMajor{});

  void *ptrDX, *ptrDY, *ptrW;
  CHECK_CUDART(
      cudaMalloc(&ptrDX, sizeof(DataTypeOutput) * cute::size(layoutDX)));
  CHECK_CUDART(
      cudaMalloc(&ptrDY, sizeof(DataTypeInput) * cute::size(layoutDY)));
  CHECK_CUDART(cudaMalloc(&ptrW, sizeof(DataTypeInput) * cute::size(layoutW)));

  auto tensorDX = std::make_shared<dllm::Tensor3D>(
      ptrDX, layoutDX, dllm::toDtype<DataTypeOutput>(), dllm::CUDA);
  auto tensorDY = std::make_shared<dllm::Tensor3D>(
      ptrDY, layoutDY, dllm::toDtype<DataTypeInput>(), dllm::CUDA);
  auto tensorW = std::make_shared<dllm::Tensor2D>(
      ptrW, layoutW, dllm::toDtype<DataTypeInput>(), dllm::CUDA);

  Eigen::Matrix<DataTypeInput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostDY(m * s, n), hostW(n, k);
  Eigen::Matrix<DataTypeOutput, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hostDX(m * s, k), refDX;

  hostDY.setRandom();
  hostW.setRandom();

  refDX =
      (hostDY.template cast<ComputeType>() * hostW.template cast<ComputeType>())
          .template cast<DataTypeOutput>();

  CHECK_CUDART(cudaMemcpy(ptrDY, hostDY.data(),
                          sizeof(DataTypeInput) * cute::size(layoutDY),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaMemcpy(ptrW, hostW.data(),
                          sizeof(DataTypeInput) * cute::size(layoutW),
                          cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  auto tensorDX2D = dllm::util::flatten<2>(tensorDX);
  auto tensorDY2D = dllm::util::flatten<2>(tensorDY);
  auto task = dllm::compute::FcNoBias::backwardX(
      tensorDX2D, tensorDY2D, tensorW, toCublasComputeType<ComputeType>());
  auto future = tensorW->future;
  tensorDX.reset();
  tensorDY.reset();
  tensorW.reset();
  threadPool.submit(std::move(task));
  {
    dllm::util::FutureGuard{future->rFuture};
    dllm::util::FutureGuard{future->wFuture};
  }

  CHECK_CUDART(cudaMemcpy(hostDX.data(), ptrDX,
                          sizeof(DataTypeOutput) * cute::size(layoutDX),
                          cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  for (int col = 0; col < refDX.cols(); ++col) {
    for (int row = 0; row < refDX.rows(); ++row) {
      ASSERT_NEAR(hostDX(row, col), refDX(row, col), 1e-4);
    }
  }

  CHECK_CUDART(cudaFree(ptrDX));
  CHECK_CUDART(cudaFree(ptrDY));
  CHECK_CUDART(cudaFree(ptrW));
}
}  // namespace

TEST_F(FcThreadPoolComputeTestFixture, TestBackwardXF16F32F32) {
  TestThreadPoolComputeBackwardXT<nv_half, float, float>(threadPool);
}
TEST_F(FcThreadPoolComputeTestFixture, TestBackwardXF32F32F32) {
  TestThreadPoolComputeBackwardXT<float, float, float>(threadPool);
}
TEST_F(FcThreadPoolComputeTestFixture, TestBackwardXF64F64F64) {
  TestThreadPoolComputeBackwardXT<double, double, double>(threadPool);
}
