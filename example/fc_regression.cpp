#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "compute/fc.h"
#include "compute/init.h"
#include "compute/mse.h"
#include "logger.h"
#include "optimizer/sgd.h"
#include "tensor.h"
#include "thread_pool.h"
#include "util.h"

void printTensor(const dllm::Tensor1D &tensor) {
  if (tensor.deviceType == dllm::CUDA) {
    switch (tensor.dtype) {
      case dllm::R_32F: {
        using T = float;
        const auto size = cute::size(tensor.layout);
        Eigen::Vector<T, Eigen::Dynamic> buffer(size);
        printf("Future: %p, valid: %d\n", tensor.future.get(), tensor.future->valid());
        dllm::util::waitFutureIfValid(tensor.future);
        CHECK_CUDART(cudaMemcpy(buffer.data(), tensor.data(), sizeof(T) * size,
                                cudaMemcpyDeviceToHost));
        CHECK_CUDART(cudaDeviceSynchronize());
        std::cout << buffer.transpose() << std::endl;
      }
    }
  }
}

int main() {
  // dtype: FP32
  // f(x) = 2x + 5
  // 1000 samples, batch size = 100, no shuffle
  // x (N x 1) -> FC1 -> f1 (N x 8) -> FC2 -> f2 (N x 1) -> MSE
  // FC1.w (8 x 1)
  // FC2.w (1 x 8)
  using T = float;
  const double lr = 1e-3;
  auto f = [](auto x) {
    using Scalar = decltype(x)::Scalar;
    return static_cast<Scalar>(2) * x.array() + static_cast<Scalar>(5);
  };
  const int inputDim = 1;
  const int batchSize = 100;
  const int sampleNum = 1000;
  using Input = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using Output = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  std::vector<std::tuple<Input, Output>> data;
  data.reserve(sampleNum / batchSize);
  for (int i = 0; i < sampleNum / batchSize; ++i) {
    Input x(batchSize, inputDim);
    x.setRandom();
    Output y = f(x);
    data.emplace_back(std::move(x), std::move(y));
  }

  std::vector<std::tuple<T *, T *>> dataDevice;
  dataDevice.reserve(data.size());
  for (auto &[x, y] : data) {
    T *input, *output;
    CHECK_CUDART(
        cudaMalloc(reinterpret_cast<void **>(&input), sizeof(T) * x.size()));
    CHECK_CUDART(cudaMemcpy(input, x.data(), sizeof(T) * x.size(),
                            cudaMemcpyHostToDevice));
    CHECK_CUDART(
        cudaMalloc(reinterpret_cast<void **>(&output), sizeof(T) * y.size()));
    CHECK_CUDART(cudaMemcpy(output, y.data(), sizeof(T) * y.size(),
                            cudaMemcpyHostToDevice));
    dataDevice.emplace_back(input, output);
  }

  using SharedTensor2D = std::shared_ptr<dllm::Tensor2D>;
  std::vector<std::tuple<SharedTensor2D, SharedTensor2D>> tensor;
  tensor.reserve(dataDevice.size());
  for (auto [ptrX, ptrY] : dataDevice) {
    auto shapeX = cute::make_shape(batchSize, inputDim);
    auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
    auto tensorX = std::make_shared<dllm::Tensor2D>(
        ptrX, layoutX, dllm::toDtype<T>(), dllm::CUDA);

    auto shapeY = cute::make_shape(batchSize, inputDim);
    auto layoutY = cute::make_layout(shapeY, cute::GenRowMajor{});
    auto tensorY = std::make_shared<dllm::Tensor2D>(
        ptrY, layoutY, dllm::toDtype<T>(), dllm::CUDA);

    tensor.emplace_back(std::move(tensorX), std::move(tensorY));
  }

  const int outDimW1 = 8;
  T *ptrW1;
  auto shapeW1 = cute::make_shape(outDimW1, inputDim);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrW1),
                          sizeof(T) * cute::size(shapeW1)));
  auto layoutW1 = cute::make_layout(shapeW1, cute::GenRowMajor{});
  auto tensorW1 = std::make_shared<dllm::Tensor2D>(
      ptrW1, layoutW1, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrDW1;
  auto shapeDW1 = cute::make_shape(outDimW1, inputDim);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrDW1),
                          sizeof(T) * cute::size(shapeDW1)));
  auto layoutDW1 = cute::make_layout(shapeDW1, cute::GenRowMajor{});
  auto tensorDW1 = std::make_shared<dllm::Tensor2D>(
      ptrDW1, layoutDW1, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrW1Out;
  auto shapeW1Out = cute::make_shape(batchSize, outDimW1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrW1Out),
                          sizeof(T) * cute::size(shapeW1Out)));
  auto layoutW1Out = cute::make_layout(shapeW1Out, cute::GenRowMajor{});
  auto tensorW1Out = std::make_shared<dllm::Tensor2D>(
      ptrW1Out, layoutW1Out, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrDW1Out;
  auto shapeDW1Out = cute::make_shape(batchSize, outDimW1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrDW1Out),
                          sizeof(T) * cute::size(shapeDW1Out)));
  auto layoutDW1Out = cute::make_layout(shapeDW1Out, cute::GenRowMajor{});
  auto tensorDW1Out = std::make_shared<dllm::Tensor2D>(
      ptrDW1Out, layoutDW1Out, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrW2;
  auto shapeW2 = cute::make_shape(1, outDimW1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrW2),
                          sizeof(T) * cute::size(shapeW2)));
  auto layoutW2 = cute::make_layout(shapeW2, cute::GenRowMajor{});
  auto tensorW2 = std::make_shared<dllm::Tensor2D>(
      ptrW2, layoutW2, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrDW2;
  auto shapeDW2 = cute::make_shape(1, outDimW1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrDW2),
                          sizeof(T) * cute::size(shapeDW2)));
  auto layoutDW2 = cute::make_layout(shapeDW2, cute::GenRowMajor{});
  auto tensorDW2 = std::make_shared<dllm::Tensor2D>(
      ptrDW2, layoutDW2, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrW2Out;
  auto shapeW2Out = cute::make_shape(batchSize, 1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrW2Out),
                          sizeof(T) * cute::size(shapeW2Out)));
  auto layoutW2Out = cute::make_layout(shapeW2Out, cute::GenRowMajor{});
  auto tensorW2Out = std::make_shared<dllm::Tensor2D>(
      ptrW2Out, layoutW2Out, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrDW2Out;
  auto shapeDW2Out = cute::make_shape(batchSize, 1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrDW2Out),
                          sizeof(T) * cute::size(shapeDW2Out)));
  auto layoutDW2Out = cute::make_layout(shapeDW2Out, cute::GenRowMajor{});
  auto tensorDW2Out = std::make_shared<dllm::Tensor2D>(
      ptrDW2Out, layoutDW2Out, dllm::toDtype<T>(), dllm::CUDA);

  dllm::ThreadPool threadPool{0, 1};

  {
    auto task = dllm::compute::Init::kaimingNorm(tensorW1);
    auto future = threadPool.submit(std::move(task));
    tensorW1->future = future;
  }
  {
    auto task = dllm::compute::Init::kaimingNorm(tensorW2);
    auto future = threadPool.submit(std::move(task));
    tensorW2->future = future;
  }

  printTensor(*dllm::util::flatten<1>(tensorW1));
  printTensor(*dllm::util::flatten<1>(tensorW2));

  const int epoch = 10;
  for (int i = 0; i < epoch; ++i) {
    for (auto &[tensorX, tensorY] : tensor) {
      {
        auto task = dllm::compute::FcNoBias::forward(
            tensorW1Out, tensorX, tensorW1, CUBLAS_COMPUTE_32F_PEDANTIC);
        auto future = threadPool.submit(std::move(task));
        tensorW1Out->future = future;
      }
      {
        auto task = dllm::compute::FcNoBias::forward(
            tensorW2Out, tensorW1Out, tensorW2, CUBLAS_COMPUTE_32F_PEDANTIC);
        auto future = threadPool.submit(std::move(task));
        tensorW2Out->future = future;
      }
      {
        auto task = dllm::compute::Mse::backward(
            dllm::util::flatten<1>(tensorDW2Out),
            dllm::util::toConstSharedPtr(dllm::util::flatten<1>(tensorW2Out)),
            dllm::util::toConstSharedPtr(dllm::util::flatten<1>(tensorY)));
        auto future = threadPool.submit(std::move(task));
        tensorDW2Out->future = future;
      }
      {
        auto task = dllm::compute::FcNoBias::backwardW(
            tensorDW2, dllm::util::toConstSharedPtr(tensorDW2Out),
            dllm::util::toConstSharedPtr(tensorW1Out),
            CUBLAS_COMPUTE_32F_PEDANTIC);
        auto future = threadPool.submit(std::move(task));
        tensorDW2->future = future;
      }
      {
        auto task = dllm::compute::FcNoBias::backwardX(
            tensorDW1Out, dllm::util::toConstSharedPtr(tensorDW2Out),
            dllm::util::toConstSharedPtr(tensorW2),
            CUBLAS_COMPUTE_32F_PEDANTIC);
        auto future = threadPool.submit(std::move(task));
        tensorDW1Out->future = future;
      }
      {
        auto task = dllm::compute::FcNoBias::backwardW(
            tensorDW1, dllm::util::toConstSharedPtr(tensorDW1Out),
            dllm::util::toConstSharedPtr(tensorW1Out),
            CUBLAS_COMPUTE_32F_PEDANTIC);
        auto future = threadPool.submit(std::move(task));
        tensorDW1->future = future;
      }
      {
        auto task = dllm::optimizer::Sgd::step(
            dllm::util::flatten<1>(tensorW2),
            dllm::util::toConstSharedPtr(dllm::util::flatten<1>(tensorDW2)),
            lr);
        auto future = threadPool.submit(std::move(task));
        tensorW2->future = future;
      }
      {
        auto task = dllm::optimizer::Sgd::step(
            dllm::util::flatten<1>(tensorW1),
            dllm::util::toConstSharedPtr(dllm::util::flatten<1>(tensorDW1)),
            lr);
        auto future = threadPool.submit(std::move(task));
        tensorW1->future = future;
      }
      tensorW2->future->wait();
      tensorW1->future->wait();
    }
  }

  Input xTest(batchSize, inputDim);
  xTest.setRandom();
  Output yTest = f(xTest);
  Output yTestRef(batchSize, 1);

  auto &[tensorX, tensorY] = tensor[0];

  CHECK_CUDART(cudaMemcpy(tensorX->data(), xTest.data(),
                          sizeof(T) * xTest.size(), cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  {
    auto task = dllm::compute::FcNoBias::forward(tensorW1Out, tensorX, tensorW1,
                                                 CUBLAS_COMPUTE_32F_PEDANTIC);
    auto future = threadPool.submit(std::move(task));
    tensorW1Out->future = future;
  }
  {
    auto task = dllm::compute::FcNoBias::forward(
        tensorW2Out, tensorW1Out, tensorW2, CUBLAS_COMPUTE_32F_PEDANTIC);
    auto future = threadPool.submit(std::move(task));
    tensorW2Out->future = future;
  }
  tensorW2Out->future->wait();

  CHECK_CUDART(cudaMemcpy(yTestRef.data(), tensorW2Out->data(),
                          sizeof(T) * yTest.size(), cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  std::cout << yTest.topRows(5).transpose() << std::endl;
  std::cout << yTestRef.topRows(5).transpose() << std::endl;
  printf("Error: %f\n", (yTest - yTestRef).norm());

  CHECK_CUDART(cudaFree(ptrW2Out));
  CHECK_CUDART(cudaFree(ptrW2));
  CHECK_CUDART(cudaFree(ptrDW1Out));
  CHECK_CUDART(cudaFree(ptrW1Out));
  CHECK_CUDART(cudaFree(ptrW1));
  for (auto [ptrX, ptrY] : dataDevice) {
    CHECK_CUDART(cudaFree(ptrX));
    CHECK_CUDART(cudaFree(ptrY));
  }
}