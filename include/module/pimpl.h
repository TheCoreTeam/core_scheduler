#pragma once

#include <torch/csrc/api/include/torch/nn/pimpl.h>

#define DLLM_API TORCH_API

/// Like `DLLM_MODULE_IMPL`, but defaults the `ImplType` name to `<Name>Impl`.
#define DLLM_MODULE(Name) TORCH_MODULE_IMPL(Name, Name##Impl)
