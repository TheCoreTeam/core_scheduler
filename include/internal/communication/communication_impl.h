#pragma once
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include "communication/communication.h"

namespace dllm::communication {
struct Comm::Impl {
  Impl(MPI_Comm group, c10::intrusive_ptr<c10d::Store> store,
       c10::intrusive_ptr<c10d::Backend> backend);

  const c10::intrusive_ptr<c10d::Store> &store() const;

  const c10::intrusive_ptr<c10d::Backend> &backend() const;

 private:
  MPI_Comm group_{};
  const c10::intrusive_ptr<c10d::Store> store_{};
  const c10::intrusive_ptr<c10d::Backend> backend_{};
};
}  // namespace dllm::communication