#include "compute/utils.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::compute::Utils {
TaskCompute range(const std::shared_ptr<Tensor> &tensor,
                  const at::Scalar &start, const at::Scalar &end,
                  const at::TensorOptions options) {
  auto task = TaskCompute{
      [tensor = tensor, start = start, end = end, options = options,
       future = tensor->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::range");
        {
          util::FutureGuard guard{future};
          DLLM_EXTRACT_TENSOR(tensor) = torch::range(start, end, options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  // size
  tensor->sizes() = IntArray{end.toLong() - start.toLong()};
  return task;
}

TaskCompute randint(const std::shared_ptr<Tensor> &tensor, const int64_t low,
                    const int64_t high, const IntArrayRef &size,
                    const at::TensorOptions options) {
  auto task = TaskCompute{
      [tensor = tensor, high = high, low = low, options = options,
       future = tensor->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::randint");
        {
          util::FutureGuard guard{future};
          DLLM_EXTRACT_TENSOR(tensor) =
              torch::randint(low, high, tensor->sizes(), options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  // size
  tensor->sizes() = size;
  return task;
}

TaskCompute empty(const std::shared_ptr<Tensor> &tensor,
                  const IntArrayRef &size, at::TensorOptions options) {
  auto task = TaskCompute{
      [tensor = tensor, options = options,
       future = tensor->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::empty");
        {
          util::FutureGuard guard{future};
          DLLM_EXTRACT_TENSOR(tensor) = torch::empty(tensor->sizes(), options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  tensor->sizes() = size;
  return task;
}

TaskCompute empty_like(const std::shared_ptr<Tensor> &dst,
                       const std::shared_ptr<const ReadOnlyTensor> &src) {
  auto task = TaskCompute{[dst = dst, src = src, dstFuture = dst->future(),
                           srcFuture = src->future()](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::empty_like");
    {
      util::FutureGuard srcGuard{srcFuture};
      util::FutureGuard dstGuard{dstFuture};
      DLLM_EXTRACT_TENSOR(dst) = torch::empty_like(DLLM_EXTRACT_TENSOR(src));
      src.reset();
      dst.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  dst->resetFuture(future);
  src->resetFuture(future);
  // size
  dst->sizes() = src->sizes();
  return task;
}

TaskCompute ones(const std::shared_ptr<Tensor> &tensor, const IntArrayRef &size,
                 at::TensorOptions options) {
  auto task = TaskCompute{
      [tensor = tensor, options = options,
       future = tensor->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::ones");
        {
          util::FutureGuard guard{future};
          DLLM_EXTRACT_TENSOR(tensor) = torch::ones(tensor->sizes(), options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  tensor->sizes() = size;
  return task;
}

TaskCompute ones_like(const std::shared_ptr<Tensor> &dst,
                      const std::shared_ptr<const ReadOnlyTensor> &src) {
  auto task = TaskCompute{
      [dst = dst, src = src, dstFuture = dst->future(),
       srcFuture = src->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::ones_like");
        {
          util::FutureGuard srcGuard{srcFuture};
          util::FutureGuard dstGuard{dstFuture};
          DLLM_EXTRACT_TENSOR(dst) = torch::ones_like(DLLM_EXTRACT_TENSOR(src));
          src.reset();
          dst.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  dst->resetFuture(future);
  src->resetFuture(future);
  // size
  dst->sizes() = src->sizes();
  return task;
}

TaskCompute zeros(const std::shared_ptr<Tensor> &tensor,
                  const IntArrayRef &size, at::TensorOptions options) {
  auto task = TaskCompute{
      [tensor = tensor, options = options,
       future = tensor->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::zeros");
        {
          util::FutureGuard guard{future};
          DLLM_EXTRACT_TENSOR(tensor) = torch::zeros(tensor->sizes(), options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  // size
  tensor->sizes() = size;
  return task;
}

TaskCompute zeros_like(const std::shared_ptr<Tensor> &dst,
                       const std::shared_ptr<const ReadOnlyTensor> &src) {
  auto task = TaskCompute{[dst = dst, src = src, dstFuture = dst->future(),
                           srcFuture = src->future()](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::zeros_like");
    {
      util::FutureGuard srcGuard{srcFuture};
      util::FutureGuard dstGuard{dstFuture};
      DLLM_EXTRACT_TENSOR(dst) = torch::zeros_like(DLLM_EXTRACT_TENSOR(src));
      src.reset();
      dst.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  dst->resetFuture(future);
  src->resetFuture(future);
  // size
  dst->sizes() = src->sizes();
  return task;
}

TaskCompute rand(const std::shared_ptr<Tensor> &tensor, const IntArrayRef &size,
                 at::TensorOptions options) {
  auto task = TaskCompute{
      [tensor = tensor, options = options,
       future = tensor->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::rand");
        {
          util::FutureGuard guard{future};
          DLLM_EXTRACT_TENSOR(tensor) = torch::rand(tensor->sizes(), options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  // size
  tensor->sizes() = size;
  return task;
}

TaskCompute rand_like(const std::shared_ptr<Tensor> &dst,
                      const std::shared_ptr<const ReadOnlyTensor> &src) {
  auto task = TaskCompute{
      [dst = dst, src = src, dstFuture = dst->future(),
       srcFuture = src->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::rand_like");
        {
          util::FutureGuard srcGuard{srcFuture};
          util::FutureGuard dstGuard{dstFuture};
          DLLM_EXTRACT_TENSOR(dst) = torch::rand_like(DLLM_EXTRACT_TENSOR(src));
          src.reset();
          dst.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  dst->resetFuture(future);
  src->resetFuture(future);
  // size
  dst->sizes() = src->sizes();
  return task;
}

TaskCompute randn(const std::shared_ptr<Tensor> &tensor,
                  const IntArrayRef &size, at::TensorOptions options) {
  auto task = TaskCompute{
      [tensor = tensor, options = options,
       future = tensor->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::randn");
        {
          util::FutureGuard guard{future};
          DLLM_EXTRACT_TENSOR(tensor) = torch::randn(tensor->sizes(), options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  // size
  tensor->sizes() = size;
  return task;
}

TaskCompute randn_like(const std::shared_ptr<Tensor> &dst,
                       const std::shared_ptr<const ReadOnlyTensor> &src) {
  auto task = TaskCompute{[dst = dst, src = src, dstFuture = dst->future(),
                           srcFuture = src->future()](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::randn_like");
    {
      util::FutureGuard srcGuard{srcFuture};
      util::FutureGuard dstGuard{dstFuture};
      DLLM_EXTRACT_TENSOR(dst) = torch::randn_like(DLLM_EXTRACT_TENSOR(src));
      src.reset();
      dst.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  dst->resetFuture(future);
  src->resetFuture(future);
  // size
  dst->sizes() = src->sizes();
  return task;
}

TaskCompute split(std::vector<std::shared_ptr<const ReadOnlyTensor>> &output,
                  const std::shared_ptr<const ReadOnlyTensor> &src,
                  const int64_t &split_size, const int64_t &dim) {
  auto input_size = src->sizes();
  const auto split_num =
      input_size[dim > 0 ? dim : input_size.size() + dim] / split_size;
  input_size[dim > 0 ? dim : input_size.size() + dim] = split_size;
  output.resize(split_num);
  at::SmallVector<std::shared_ptr<Tensor>> outputNew;
  outputNew.reserve(split_size);
  auto futurePtr = TensorFriend::extract_future_ptr(src);
  for (auto &p : output) {
    auto pNew = TensorFriend::create(futurePtr);
    // size
    pNew->sizes() = input_size;
    outputNew.push_back(pNew);
    p = std::move(pNew);
  }
  const auto p0 = outputNew[0];
  auto task = TaskCompute{[output = std::move(outputNew), src = src,
                           srcFuture = src->future(), split_size = split_size,
                           dim = dim](const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::split");
    {
      util::FutureGuard guard{srcFuture};
      const auto v = DLLM_EXTRACT_TENSOR(src).split(split_size, dim);
      for (std::size_t i = 0; i < v.size(); ++i) {
        DLLM_EXTRACT_TENSOR(output[i]) = v[i];
        output[i].reset();
      }
      src.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};

  p0->resetFuture(task.get_future());
  return task;
}

TaskCompute view(std::shared_ptr<const ReadOnlyTensor> &output,
                 const std::shared_ptr<const ReadOnlyTensor> &input,
                 const IntArrayRef &size) {
  auto outputNew =
      TensorFriend::create(TensorFriend::extract_future_ptr(input));
  auto task = TaskCompute{
      [output = outputNew, input = input,
       inputFuture = input->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::view");
        {
          util::FutureGuard dstGuard{inputFuture};
          DLLM_EXTRACT_TENSOR(output) =
              DLLM_EXTRACT_TENSOR(input).view(output->sizes());
          input.reset();
          output.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  outputNew->resetFuture(task.get_future());
  // size
  outputNew->sizes() = size;
  output = std::move(outputNew);
  return task;
}

TaskCompute broadcast_to(std::shared_ptr<const ReadOnlyTensor> &output,
                         const std::shared_ptr<const ReadOnlyTensor> &input,
                         const IntArrayRef &size) {
  auto outputNew =
      TensorFriend::create(TensorFriend::extract_future_ptr(input));
  auto task = TaskCompute{
      [output = outputNew, input = input,
       inputFuture = input->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::broadcast");
        {
          util::FutureGuard dstGuard{inputFuture};
          DLLM_EXTRACT_TENSOR(output) =
              DLLM_EXTRACT_TENSOR(input).broadcast_to(output->sizes());
          input.reset();
          output.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  outputNew->resetFuture(task.get_future());
  // size
  outputNew->sizes() = size;
  output = std::move(outputNew);
  return task;
}
}  // namespace dllm::compute::Utils
