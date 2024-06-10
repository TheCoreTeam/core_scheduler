#include "threading/task.h"

#include "threading/task_impl.h"

namespace dllm {
Task::Task(std::shared_ptr<Impl> impl) : impl_{std::move(impl)} {}

const std::vector<ReadOnlyTensor>& Task::input() const {
  return impl_->input();
}

const std::vector<Tensor>& Task::output() const { return impl_->output(); }

void Task::operator()() const { impl_->operator()(); }

const std::shared_ptr<Task::Impl>& Task::impl() const { return impl_; }

void Task::reset() { impl_.reset(); }

bool Task::valid() const { return impl_ != nullptr; }

const char* Task::name() const { return impl_->name(); }
}  // namespace dllm
