#pragma once

#include <torch/csrc/api/include/torch/nn/pimpl.h>

#define DLLM_API TORCH_API

#define DLLM_MODULE_IMPL(Name, ImplType)                               \
  class Name : public torch::nn::ModuleHolder<ImplType> { /* NOLINT */ \
   public:                                                             \
    using torch::nn::ModuleHolder<ImplType>::ModuleHolder;             \
    using Impl TORCH_UNUSED_EXCEPT_CUDA = ImplType;                    \
    using Options = Impl::Options;                                     \
  }

/// Like `DLLM_MODULE_IMPL`, but defaults the `ImplType` name to `<Name>Impl`.
#define DLLM_MODULE(Name) DLLM_MODULE_IMPL(Name, Name##Impl)
