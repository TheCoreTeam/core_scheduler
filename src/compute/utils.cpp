#include "compute/utils.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"

namespace dllm::compute::Utils {
void sum(const Scheduler &scheduler, Tensor &output,
         const ReadOnlyTensor &input, const IntArray &dim, const bool keep_dim,
         c10::optional<at::ScalarType> dtype) {
  Tensor output_{};
  DLLM_ASSERT_TRUE(!dim.empty(), "dim should not be empty");
  auto task = TaskCompute{[output = output_, input = input, dim = dim,
                           keep_dim = keep_dim, dtype = dtype,
                           inputFuture = utils::future(input)](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::range");
    {
      utils::FutureGuard inputGuard{inputFuture};
      output.impl()->tensor() =
          at::sum(input.impl()->tensor(), dim, keep_dim, dtype);
      output.reset();
      input.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(input, future);
  utils::resetFuture(output_, future);
  // size
  output_.sizes() = [&, dim = dim]() mutable {
    auto size = input.sizes();
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
  output = output_;
  scheduler.impl()->submit(std::move(task));
}

void range(const Scheduler &scheduler, Tensor &tensor, const at::Scalar &start,
           const at::Scalar &end, const at::TensorOptions options) {
  tensor = Tensor{};
  auto task =
      TaskCompute{[tensor = tensor, start = start, end = end,
                   options = options](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::range");
        {
          tensor.impl()->tensor() = torch::range(start, end, options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(tensor, future);
  // size
  tensor.sizes() = IntArray{end.toLong() - start.toLong()};
  scheduler.impl()->submit(std::move(task));
}

void arange(const Scheduler &scheduler, Tensor &tensor, const at::Scalar &start,
            const at::Scalar &end, const at::TensorOptions options) {
  tensor = Tensor{};
  auto task =
      TaskCompute{[tensor = tensor, start = start, end = end,
                   options = options](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::range");
        {
          tensor.impl()->tensor() = torch::arange(start, end, options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(tensor, future);
  // size
  tensor.sizes() = IntArray{(end.toLong() - start.toLong())};
  tensor.options() = options;
  scheduler.impl()->submit(std::move(task));
}

void arange(const Scheduler &scheduler, Tensor &tensor, const at::Scalar &start,
            const at::Scalar &end, const at::Scalar &step,
            const at::TensorOptions options) {
  tensor = Tensor{};
  auto task =
      TaskCompute{[tensor = tensor, start = start, end = end, step = step,
                   options = options](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::range");
        {
          tensor.impl()->tensor() = torch::arange(start, end, step, options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(tensor, future);
  // size
  tensor.sizes() = IntArray{(end.toLong() - start.toLong()) / step.toLong()};
  tensor.options() = options;
  scheduler.impl()->submit(std::move(task));
}

void randint(const Scheduler &scheduler, Tensor &tensor, const int64_t low,
             const int64_t high, const IntArrayRef &size,
             const at::TensorOptions options) {
  tensor = Tensor{};
  auto task =
      TaskCompute{[tensor = tensor, high = high, low = low,
                   options = options](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::randint");
        {
          tensor.impl()->tensor() =
              torch::randint(low, high, tensor.sizes(), options);
          tensor.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(tensor, future);
  // size
  tensor.sizes() = size;
  tensor.options() = options;
  scheduler.impl()->submit(std::move(task));
}

void empty(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
           at::TensorOptions options) {
  tensor = Tensor{};
  auto task = TaskCompute{[tensor = tensor, options = options](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::empty");
    {
      tensor.impl()->tensor() = torch::empty(tensor.sizes(), options);
      tensor.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(tensor, future);
  // size
  tensor.sizes() = size;
  tensor.options() = options;
  scheduler.impl()->submit(std::move(task));
}

void empty_like(const Scheduler &scheduler, Tensor &dst,
                const ReadOnlyTensor &src) {
  Tensor dst_{};
  auto task =
      TaskCompute{[dst = dst_, src = src, srcFuture = utils::future(src)](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::empty_like");
        {
          utils::FutureGuard srcGuard{srcFuture};
          dst.impl()->tensor() = torch::empty_like(src.impl()->tensor());
          src.reset();
          dst.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(dst_, future);
  utils::resetFuture(src, future);
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  dst = dst_;
  scheduler.impl()->submit(std::move(task));
}

void ones(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
          at::TensorOptions options) {
  tensor = Tensor{};
  auto task = TaskCompute{[tensor = tensor, options = options](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::ones");
    {
      tensor.impl()->tensor() = torch::ones(tensor.sizes(), options);
      tensor.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(tensor, future);
  tensor.sizes() = size;
  tensor.options() = options;
  scheduler.impl()->submit(std::move(task));
}

void ones_like(const Scheduler &scheduler, Tensor &dst,
               const ReadOnlyTensor &src) {
  Tensor dst_{};
  auto task =
      TaskCompute{[dst = dst_, src = src, srcFuture = utils::future(src)](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::ones_like");
        {
          utils::FutureGuard srcGuard{srcFuture};
          dst.impl()->tensor() = torch::ones_like(src.impl()->tensor());
          src.reset();
          dst.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(dst_, future);
  utils::resetFuture(src, future);
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  dst = dst_;
  scheduler.impl()->submit(std::move(task));
}

void zeros(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
           at::TensorOptions options) {
  tensor = Tensor{};
  auto task = TaskCompute{[tensor = tensor, options = options](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::zeros");
    {
      tensor.impl()->tensor() = torch::zeros(tensor.sizes(), options);
      tensor.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(tensor, future);
  // size
  tensor.sizes() = size;
  tensor.options() = options;
  scheduler.impl()->submit(std::move(task));
}

void zeros_like(const Scheduler &scheduler, Tensor &dst,
                const ReadOnlyTensor &src) {
  Tensor dst_{};
  auto task =
      TaskCompute{[dst = dst_, src = src, srcFuture = utils::future(src)](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::zeros_like");
        {
          utils::FutureGuard srcGuard{srcFuture};
          dst.impl()->tensor() = torch::zeros_like(src.impl()->tensor());
          src.reset();
          dst.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(dst_, future);
  utils::resetFuture(src, future);
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  dst = dst_;
  scheduler.impl()->submit(std::move(task));
}

void rand(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
          at::TensorOptions options) {
  tensor = Tensor{};
  auto task = TaskCompute{[tensor = tensor, options = options](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::rand");
    {
      tensor.impl()->tensor() = torch::rand(tensor.sizes(), options);
      tensor.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(tensor, future);
  // size
  tensor.sizes() = size;
  tensor.options() = options;
  scheduler.impl()->submit(std::move(task));
}

void rand_like(const Scheduler &scheduler, Tensor &dst,
               const ReadOnlyTensor &src) {
  Tensor dst_{};
  auto task =
      TaskCompute{[dst = dst_, src = src, srcFuture = utils::future(src)](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::rand_like");
        {
          utils::FutureGuard srcGuard{srcFuture};
          dst.impl()->tensor() = torch::rand_like(src.impl()->tensor());
          src.reset();
          dst.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(dst_, future);
  utils::resetFuture(src, future);
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  dst = dst_;
  scheduler.impl()->submit(std::move(task));
}

void randn(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
           at::TensorOptions options) {
  tensor = Tensor{};
  auto task = TaskCompute{[tensor = tensor, options = options](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::randn");
    {
      tensor.impl()->tensor() = torch::randn(tensor.sizes(), options);
      tensor.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(tensor, future);
  // size
  tensor.sizes() = size;
  tensor.options() = options;
  scheduler.impl()->submit(std::move(task));
}

void randn_like(const Scheduler &scheduler, Tensor &dst,
                const ReadOnlyTensor &src) {
  Tensor dst_{};
  auto task =
      TaskCompute{[dst = dst_, src = src, srcFuture = utils::future(src)](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::randn_like");
        {
          utils::FutureGuard srcGuard{srcFuture};
          dst.impl()->tensor() = torch::randn_like(src.impl()->tensor());
          src.reset();
          dst.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(dst_, future);
  utils::resetFuture(src, future);
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  dst = dst_;
  scheduler.impl()->submit(std::move(task));
}

void split(const Scheduler &scheduler, std::vector<ReadOnlyTensor> &output,
           const ReadOnlyTensor &src, const int64_t &split_size,
           const int64_t &dim) {
  auto input_size = src.sizes();
  const auto split_num =
      input_size[dim > 0 ? dim : input_size.size() + dim] / split_size;
  input_size[dim > 0 ? dim : input_size.size() + dim] = split_size;
  output.resize(split_num);
  at::SmallVector<Tensor> outputNew;
  outputNew.reserve(split_size);
  const auto futurePtr = src.impl()->futurePtr();
  for (auto &p : output) {
    Tensor pNew;
    pNew.impl()->futurePtr() = futurePtr;
    // size
    pNew.sizes() = input_size;
    pNew.options() = src.options();
    outputNew.push_back(pNew);
    p = std::move(pNew);
  }
  const auto p0 = outputNew[0];
  auto task =
      TaskCompute{[output = std::move(outputNew), src = src,
                   srcFuture = utils::future(src), split_size = split_size,
                   dim = dim](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::split");
        {
          utils::FutureGuard guard{srcFuture};
          const auto v = src.impl()->tensor().split(split_size, dim);
          for (std::size_t i = 0; i < v.size(); ++i) {
            output[i].impl()->tensor() = v[i];
            output[i].reset();
          }
          src.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};

  utils::resetFuture(p0, task.get_future());
  scheduler.impl()->submit(std::move(task));
}

void view(const Scheduler &scheduler, Tensor &output,
          const ReadOnlyTensor &input, const IntArrayRef &size) {
  Tensor output_{};
  auto task = TaskCompute{
      [output = output_, input = input, inputFuture = utils::future(input)](
          const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::view");
        {
          utils::FutureGuard inputGuard{inputFuture};
          output.impl()->tensor() = input.impl()->tensor().view(output.sizes());
          input.reset();
          output.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(input, future);
  utils::resetFuture(output_, future);
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
  output_.sizes() = toNewShape(input.sizes(), size);
  output_.options() = input.options();
  output_.impl()->futurePtr() = input.impl()->futurePtr();
  output = output_;
  scheduler.impl()->submit(std::move(task));
}

void broadcast_to(const Scheduler &scheduler, Tensor &output,
                  const ReadOnlyTensor &input, const IntArrayRef &size) {
  Tensor output_{};
  auto task = TaskCompute{
      [output = output_, input = input, inputFuture = utils::future(input)](
          const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Utils::broadcast_to");
        {
          utils::FutureGuard inputGuard{inputFuture};
          output.impl()->tensor() =
              input.impl()->tensor().broadcast_to(output.sizes());
          input.reset();
          output.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(input, future);
  utils::resetFuture(output, future);
  // size
  output_.sizes() = size;
  output_.options() = input.options();
  output_.impl()->futurePtr() = input.impl()->futurePtr();
  output = output_;
  scheduler.impl()->submit(std::move(task));
}

void cat(const Scheduler &scheduler, Tensor &output,
         const std::vector<ReadOnlyTensor> &input, const int64_t dim) {
  output = Tensor{};
  std::vector<TaskFuture> inputFuture;
  inputFuture.reserve(input.size());
  for (const auto &t : input) {
    inputFuture.push_back(utils::future(t));
  }
  auto task = TaskCompute{[dim = dim, output = output, input = input,
                           inputFuture = std::move(inputFuture)](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Utils::cat");
    {
      for (auto &f : inputFuture) {
        utils::FutureGuard{f};
      }
      std::vector<at::Tensor> vInput;
      vInput.reserve(input.size());
      for (const auto &t : input) {
        vInput.push_back(t.impl()->tensor());
      }
      output.impl()->tensor() = at::cat(vInput, dim);
      output.reset();
      input.clear();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(output, future);
  for (const auto &t : input) {
    utils::resetFuture(t, future);
  }
  // size
  output.sizes() = [&] {
    auto size = input[0].sizes();
    const auto positiveDim =
        dim < 0 ? static_cast<int64_t>(size.size()) + dim : dim;
    size[positiveDim] = 0;
    for (const auto &t : input) {
      size[positiveDim] += t.size(positiveDim);
    }
    return size;
  }();
  output.options() = input[0].options();
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::compute::Utils
