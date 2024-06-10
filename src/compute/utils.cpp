#include "compute/utils.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"
#include "threading/task_impl.h"

namespace dllm::compute::Utils {
void sum(const Scheduler &scheduler, Tensor &output,
         const ReadOnlyTensor &input, const IntArray &dim, const bool keep_dim,
         const c10::optional<at::ScalarType> dtype) {
  struct Impl : Task::Impl {
    const IntArray dim;
    const bool keep_dim;
    const c10::optional<at::ScalarType> dtype;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */,
                  const IntArray &dim, const bool keep_dim,
                  const c10::optional<at::ScalarType> dtype)
        : Task::Impl{std::move(output), std::move(input), compute},
          dim{dim},
          keep_dim{keep_dim},
          dtype{dtype} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          at::sum(input()[0].impl()->tensor(), dim, keep_dim, dtype);
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::sum";
    }
  };

  Tensor output_{};
  DLLM_ASSERT_TRUE(!dim.empty(), "dim should not be empty");
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

  scheduler.impl()->submit(Task{
      std::make_shared<Impl>(Impl{{output_}, {input}, dim, keep_dim, dtype})});
  output = output_;
}

void range(const Scheduler &scheduler, Tensor &tensor, const at::Scalar &start,
           const at::Scalar &end, const TensorOptions options) {
  struct Impl : Task::Impl {
    const at::Scalar start, end;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const at::Scalar &start, const at::Scalar &end,
                  const TensorOptions options)
        : Task::Impl{std::move(output), {}, compute},
          start{start},
          end{end},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::range(start, end, options);
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::range";
    }
  };

  tensor = Tensor{};
  // size
  tensor.sizes() = IntArray{end.toLong() - start.toLong()};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, start, end, options})});
}

void arange(const Scheduler &scheduler, Tensor &tensor, const at::Scalar &start,
            const at::Scalar &end, const TensorOptions options) {
  struct Impl : Task::Impl {
    const at::Scalar start, end;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const at::Scalar &start, const at::Scalar &end,
                  const TensorOptions options)
        : Task::Impl{std::move(output), {}, compute},
          start{start},
          end{end},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::arange(start, end, options);
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::arange";
    }
  };

  tensor = Tensor{};
  // size
  tensor.sizes() = IntArray{(end.toLong() - start.toLong())};
  tensor.options() = options;
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, start, end, options})});
}

void arange(const Scheduler &scheduler, Tensor &tensor, const at::Scalar &start,
            const at::Scalar &end, const at::Scalar &step,
            const TensorOptions options) {
  struct Impl : Task::Impl {
    const at::Scalar start, end, step;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const at::Scalar &start, const at::Scalar &end,
                  const at::Scalar &step, const TensorOptions options)
        : Task::Impl{std::move(output), {}, compute},
          start{start},
          end{end},
          step{step},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::arange(start, end, step, options);
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::arange";
    }
  };

  tensor = Tensor{};
  // size
  tensor.sizes() = IntArray{(end.toLong() - start.toLong()) / step.toLong()};
  tensor.options() = options;
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, start, end, step, options})});
}

void randint(const Scheduler &scheduler, Tensor &tensor, const int64_t low,
             const int64_t high, const IntArrayRef &size,
             const TensorOptions options) {
  struct Impl : Task::Impl {
    const int64_t low, high;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */, const int64_t low,
                  const int64_t high, const TensorOptions &options)
        : Task::Impl{std::move(output), {}, compute},
          low{low},
          high{high},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::randint(low, high, output()[0].sizes(), options);
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::randint";
    }
  };

  tensor = Tensor{};
  // size
  tensor.sizes() = size;
  tensor.options() = options;
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, low, high, options})});
}

auto empty(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
           const TensorOptions options) -> void {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */)
        : Task::Impl{std::move(output), {}, compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::empty(output()[0].sizes(), output()[0].options());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::empty";
    }
  };

  Tensor tensor_{};
  // size
  tensor_.sizes() = size;
  tensor_.options() = options;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{tensor_}})});
  tensor = tensor_;
}

void empty_like(const Scheduler &scheduler, Tensor &dst,
                const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::empty_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::empty_like";
    }
  };

  Tensor dst_{};
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst_}, {src}})});
  dst = dst_;
}

void ones(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
          const TensorOptions options) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */)
        : Task::Impl{std::move(output), {}, compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::ones(output()[0].sizes(), output()[0].options());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::ones";
    }
  };

  Tensor tensor_{};
  // size
  tensor_.sizes() = size;
  tensor_.options() = options;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{tensor_}})});
  tensor = tensor_;
}

void ones_like(const Scheduler &scheduler, Tensor &dst,
               const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::ones_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::ones_like";
    }
  };

  Tensor dst_{};
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst_}, {src}})});
  dst = dst_;
}

void zeros(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
           const TensorOptions options) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */)
        : Task::Impl{std::move(output), {}, compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::zeros(output()[0].sizes(), output()[0].options());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::zeros";
    }
  };

  Tensor tensor_{};
  // size
  tensor_.sizes() = size;
  tensor_.options() = options;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{tensor_}})});
  tensor = tensor_;
}

void zeros_like(const Scheduler &scheduler, Tensor &dst,
                const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::zeros_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::zeros_like";
    }
  };

  Tensor dst_{};
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst_}, {src}})});
  dst = dst_;
}

void rand(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
          const TensorOptions options) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */)
        : Task::Impl{std::move(output), {}, compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::rand(output()[0].sizes(), output()[0].options());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::rand";
    }
  };

  Tensor tensor_{};
  // size
  tensor_.sizes() = size;
  tensor_.options() = options;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{tensor_}})});
  tensor = tensor_;
}

void rand_like(const Scheduler &scheduler, Tensor &dst,
               const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::rand_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::rand_like";
    }
  };

  Tensor dst_{};
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst_}, {src}})});
  dst = dst_;
}

void randn(const Scheduler &scheduler, Tensor &tensor, const IntArrayRef &size,
           const TensorOptions options) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */)
        : Task::Impl{std::move(output), {}, compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::randn(output()[0].sizes(), output()[0].options());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::randn";
    }
  };

  Tensor tensor_{};
  // size
  tensor_.sizes() = size;
  tensor_.options() = options;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{tensor_}})});
  tensor = tensor_;
}

void randn_like(const Scheduler &scheduler, Tensor &dst,
                const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::randn_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::randn_like";
    }
  };

  Tensor dst_{};
  // size
  dst_.sizes() = src.sizes();
  dst_.options() = src.options();
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst_}, {src}})});
  dst = dst_;
}

void split(const Scheduler &scheduler, std::vector<Tensor> &output,
           const ReadOnlyTensor &src, const int64_t &split_size,
           const int64_t &dim) {
  struct Impl : Task::Impl {
    const int64_t split_size;
    const int64_t dim;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* inputs */,
                  const int64_t split_size, const int64_t dim)
        : Task::Impl{std::move(output), std::move(input), compute},
          split_size{split_size},
          dim{dim} {}
    void operator()() const override {
      const auto v = input()[0].impl()->tensor().split(split_size, dim);
      for (std::size_t i = 0; i < v.size(); ++i) {
        output()[i].impl()->tensor() = v[i];
      }
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::split";
    }
  };

  auto input_size = src.sizes();
  const auto split_num =
      input_size[dim > 0 ? dim : input_size.size() + dim] / split_size;
  input_size[dim > 0 ? dim : input_size.size() + dim] = split_size;
  output.resize(split_num);
  at::SmallVector<Tensor> outputNew;
  outputNew.reserve(split_size);
  for (auto &p : output) {
    Tensor pNew;
    // size
    pNew.sizes() = input_size;
    pNew.options() = src.options();
    outputNew.push_back(pNew);
    p = std::move(pNew);
  }
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{output, {src}, split_size, dim})});
}

void view(const Scheduler &scheduler, Tensor &output,
          const ReadOnlyTensor &input, const IntArrayRef &size) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          input()[0].impl()->tensor().view(output()[0].sizes());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::view";
    }
  };

  Tensor output_{};

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
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output_}, {input}})});
  output = output_;
}

void broadcast_to(const Scheduler &scheduler, Tensor &output,
                  const ReadOnlyTensor &input, const IntArrayRef &size) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* inputs */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          input()[0].impl()->tensor().broadcast_to(output()[0].sizes());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::broadcast_to";
    }
  };

  Tensor output_{};
  // size
  output_.sizes() = size;
  output_.options() = input.options();
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output_}, {input}})});
  output = output_;
}

void cat(const Scheduler &scheduler, Tensor &output,
         const std::vector<ReadOnlyTensor> &input, const int64_t dim) {
  struct Impl : Task::Impl {
    const int64_t dim;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* inputs */,
                  const int64_t dim)
        : Task::Impl{std::move(output), std::move(input), compute}, dim{dim} {}
    void operator()() const override {
      std::vector<at::Tensor> vInput;
      vInput.reserve(input().size());
      for (const auto &t : input()) {
        vInput.push_back(t.impl()->tensor());
      }
      output()[0].impl()->tensor() = at::cat(vInput, dim);
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Utils::cat";
    }
  };

  Tensor output_{};

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
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output_}, input, dim})});
  output = output_;
}
}  // namespace dllm::compute::Utils
