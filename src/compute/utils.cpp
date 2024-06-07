#include "compute/utils.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"

namespace dllm::compute::Utils {
void sum(const Scheduler &scheduler, const std::shared_ptr<Tensor> &output,
         const std::shared_ptr<const ReadOnlyTensor> &input,
         const IntArray &dim, const bool keep_dim,
         c10::optional<at::ScalarType> dtype) {
  DLLM_ASSERT_TRUE(!dim.empty(), "dim should not be empty");
  auto task = TaskCompute{
      [output = output, input = input, dim = dim, keep_dim = keep_dim,
       dtype = dtype, outputFuture = output->future(),
       inputFuture = input->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::range");
        {
          util::FutureGuard outputGuard{outputFuture};
          util::FutureGuard inputGuard{inputFuture};
          DLLM_EXTRACT_TENSOR(output) =
              at::sum(DLLM_EXTRACT_TENSOR(input), dim, keep_dim, dtype);
          output.reset();
          input.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  input->resetFuture(future);
  output->resetFuture(future);
  // size
  output->sizes() = [&, dim = dim]() mutable {
    auto size = input->sizes();
    const auto len = size.size();
    for (auto &d : dim) {
      if (d < 0) {
        d += len;
      }
      size[d] = 1;
    }
    if (!keep_dim) {
      std::ranges::sort(dim);
      IntArray sizeNew;
      std::size_t low = 0;
      for (const std::size_t high : dim) {
        for (std::size_t j = low; j < high; ++j) {
          sizeNew.push_back(size[j]);
        }
        low = high + 1;
      }
      for (std::size_t i = low; i < size.size(); ++i) {
        sizeNew.push_back(static_cast<int64_t>(i));
      }
      return sizeNew;
    }
    return size;
  }();
  scheduler.impl()->submit(std::move(task));
}

void range(const Scheduler &scheduler, const std::shared_ptr<Tensor> &tensor,
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
  scheduler.impl()->submit(std::move(task));
}

void arange(const Scheduler &scheduler, const std::shared_ptr<Tensor> &tensor,
            const at::Scalar &start, const at::Scalar &end,
            const at::TensorOptions options) {
  auto task = TaskCompute{
      [tensor = tensor, start = start, end = end, options = options,
       future = tensor->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::range");
        {
          util::FutureGuard guard{future};
          DLLM_EXTRACT_TENSOR(tensor) = torch::arange(start, end, options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  // size
  tensor->sizes() = IntArray{(end.toLong() - start.toLong())};
  tensor->options() = options;
  scheduler.impl()->submit(std::move(task));
}

void arange(const Scheduler &scheduler, const std::shared_ptr<Tensor> &tensor,
            const at::Scalar &start, const at::Scalar &end,
            const at::Scalar &step, const at::TensorOptions options) {
  auto task = TaskCompute{[tensor = tensor, start = start, end = end,
                           step = step, options = options,
                           future = tensor->future()](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::range");
    {
      util::FutureGuard guard{future};
      DLLM_EXTRACT_TENSOR(tensor) = torch::arange(start, end, step, options);
      tensor.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  // size
  tensor->sizes() = IntArray{(end.toLong() - start.toLong()) / step.toLong()};
  tensor->options() = options;
  scheduler.impl()->submit(std::move(task));
}

void randint(const Scheduler &scheduler, const std::shared_ptr<Tensor> &tensor,
             const int64_t low, const int64_t high, const IntArrayRef &size,
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
  tensor->options() = options;
  scheduler.impl()->submit(std::move(task));
}

void empty(const Scheduler &scheduler, const std::shared_ptr<Tensor> &tensor,
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
  // size
  tensor->sizes() = size;
  tensor->options() = options;
  scheduler.impl()->submit(std::move(task));
}

void empty_like(const Scheduler &scheduler, const std::shared_ptr<Tensor> &dst,
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
  dst->options() = src->options();
  scheduler.impl()->submit(std::move(task));
}

void ones(const Scheduler &scheduler, const std::shared_ptr<Tensor> &tensor,
          const IntArrayRef &size, at::TensorOptions options) {
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
  tensor->options() = options;
  scheduler.impl()->submit(std::move(task));
}

void ones_like(const Scheduler &scheduler, const std::shared_ptr<Tensor> &dst,
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
  dst->options() = src->options();
  scheduler.impl()->submit(std::move(task));
}

void zeros(const Scheduler &scheduler, const std::shared_ptr<Tensor> &tensor,
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
  tensor->options() = options;
  scheduler.impl()->submit(std::move(task));
}

void zeros_like(const Scheduler &scheduler, const std::shared_ptr<Tensor> &dst,
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
  dst->options() = src->options();
  scheduler.impl()->submit(std::move(task));
}

void rand(const Scheduler &scheduler, const std::shared_ptr<Tensor> &tensor,
          const IntArrayRef &size, at::TensorOptions options) {
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
  tensor->options() = options;
  scheduler.impl()->submit(std::move(task));
}

void rand_like(const Scheduler &scheduler, const std::shared_ptr<Tensor> &dst,
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
  dst->options() = src->options();
  scheduler.impl()->submit(std::move(task));
}

void randn(const Scheduler &scheduler, const std::shared_ptr<Tensor> &tensor,
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
  tensor->options() = options;
  scheduler.impl()->submit(std::move(task));
}

void randn_like(const Scheduler &scheduler, const std::shared_ptr<Tensor> &dst,
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
  dst->options() = src->options();
  scheduler.impl()->submit(std::move(task));
}

void split(const Scheduler &scheduler,
           std::vector<std::shared_ptr<const ReadOnlyTensor>> &output,
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
    pNew->options() = src->options();
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
  scheduler.impl()->submit(std::move(task));
}

void view(const Scheduler &scheduler, const std::shared_ptr<Tensor> &output,
          const std::shared_ptr<const ReadOnlyTensor> &input,
          const IntArrayRef &size) {
  TensorFriend::extract_future_ptr(output) =
      TensorFriend::extract_future_ptr(input);
  auto task = TaskCompute{
      [output = output, input = input, inputFuture = input->future(),
       outputFuture = output->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::view");
        {
          util::FutureGuard inputGuard{inputFuture};
          util::FutureGuard outputGuard{outputFuture};
          DLLM_EXTRACT_TENSOR(output) =
              DLLM_EXTRACT_TENSOR(input).view(output->sizes());
          input.reset();
          output.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  input->resetFuture(future);
  output->resetFuture(future);
  // size
  auto toNewShape = [&](const auto &originalShape, const auto &newShape) {
    const int64_t totalElements =
        std::accumulate(originalShape.begin(), originalShape.end(), 1LL,
                        std::multiplies<int64_t>{});

    int64_t productOfKnownDims = 1;
    int unknownDimIndex = -1;
    for (size_t i = 0; i < newShape.size(); ++i) {
      if (newShape[i] == -1) {
        if (unknownDimIndex != -1) {
          DLLM_ASSERT_TRUE(false,
                           "More than one unknown dimension (-1) specified.");
        }
        unknownDimIndex = i;
      } else {
        productOfKnownDims *= newShape[i];
      }
    }

    IntArray resolvedShape(newShape.begin(), newShape.end());

    if (unknownDimIndex != -1) {
      const int64_t inferredDim = totalElements / productOfKnownDims;
      DLLM_ASSERT_TRUE(
          totalElements % productOfKnownDims == 0,
          "Invalid shape: total size of new array must be unchanged.");
      resolvedShape[unknownDimIndex] = inferredDim;
    } else {
      DLLM_ASSERT_TRUE(
          productOfKnownDims == totalElements,
          "Invalid shape: total size of new array must be unchanged.");
    }

    return resolvedShape;
  };
  output->sizes() = toNewShape(input->sizes(), size);
  output->options() = input->options();
  scheduler.impl()->submit(std::move(task));
}

void broadcast_to(const Scheduler &scheduler,
                  const std::shared_ptr<Tensor> &output,
                  const std::shared_ptr<const ReadOnlyTensor> &input,
                  const IntArrayRef &size) {
  TensorFriend::extract_future_ptr(output) =
      TensorFriend::extract_future_ptr(input);
  auto task = TaskCompute{
      [output = output, input = input, inputFuture = input->future(),
       outputFuture = output->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::broadcast_to");
        {
          util::FutureGuard inputGuard{inputFuture};
          util::FutureGuard outputGuard{outputFuture};
          DLLM_EXTRACT_TENSOR(output) =
              DLLM_EXTRACT_TENSOR(input).broadcast_to(output->sizes());
          input.reset();
          output.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  input->resetFuture(future);
  output->resetFuture(future);
  // size
  output->sizes() = size;
  output->options() = input->options();
  scheduler.impl()->submit(std::move(task));
}

void cat(const Scheduler &scheduler, const std::shared_ptr<Tensor> &output,
         const std::vector<std::shared_ptr<const ReadOnlyTensor>> &input,
         const int64_t dim) {
  std::vector<TaskFuture> inputFuture;
  inputFuture.reserve(input.size());
  for (const auto &t : input) {
    inputFuture.push_back(t->future());
  }
  auto task = TaskCompute{[dim = dim, output = output, input = input,
                           outputFuture = output->future(),
                           inputFuture = std::move(inputFuture)](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::cat");
    {
      for (auto &f : inputFuture) {
        util::FutureGuard{f};
      }
      util::FutureGuard dstGuard{outputFuture};
      std::vector<at::Tensor> vInput;
      vInput.reserve(input.size());
      for (const auto &t : input) {
        vInput.push_back(DLLM_EXTRACT_TENSOR(t));
      }
      DLLM_EXTRACT_TENSOR(output) = at::cat(vInput, dim);
      output.reset();
      input.clear();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  output->resetFuture(future);
  for (const auto &t : input) {
    t->resetFuture(future);
  }
  // size
  output->sizes() = [&] {
    auto size = input[0]->sizes();
    const auto positiveDim =
        dim < 0 ? static_cast<int64_t>(size.size()) + dim : dim;
    size[positiveDim] = 0;
    for (const auto &t : input) {
      size[positiveDim] += t->size(positiveDim);
    }
    return size;
  }();
  output->options() = input[0]->options();
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::compute::Utils
