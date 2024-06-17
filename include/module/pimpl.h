/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
