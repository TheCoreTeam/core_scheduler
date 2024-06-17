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
#include <c10/util/Exception.h>
#include <torch/nn/pimpl.h>
#include <torch/ordered_dict.h>

#include <memory>
#include <string>

#include "logger.h"
#include "module/state.h"

namespace dllm::module {
struct Module : std::enable_shared_from_this<Module> {
  virtual ~Module() = default;
  using NamedModulePointerApplyFunction =
      std::function<void(const std::string&, const std::shared_ptr<Module>&)>;
  using ConstNamedModuleApplyFunction =
      std::function<void(const std::string&, const Module&)>;

  void apply_to_submodules(
      const NamedModulePointerApplyFunction& function,
      const std::string& name_prefix = std::string()) const;

  std::shared_ptr<Module> shared_from_this_checked() const;

  void apply(const ConstNamedModuleApplyFunction& function,
             const std::string& name_prefix = {}) const;

  void register_state(std::string name, std::shared_ptr<State> state);

  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name, std::shared_ptr<ModuleType> module);

  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name, torch::nn::ModuleHolder<ModuleType> module_holder);

  OrderedDict<std::string, std::shared_ptr<State>> named_states(
      bool recurse = true) const;

  OrderedDict<std::string, Tensor> named_parameters(bool recurse = true) const;

  void to(TensorOptions options) const;

 protected:
  OrderedDict<std::string, std::shared_ptr<Module>> children_;

  OrderedDict<std::string, std::shared_ptr<State>> states_;
};

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name, std::shared_ptr<ModuleType> module) {
  DLLM_ASSERT_TRUE(!name.empty(), "Submodule name must not be empty");
  DLLM_ASSERT_TRUE(name.find('.') == std::string::npos,
                   "Submodule name must not contain a dot (got '", name, "')");
  auto& base_module = children_.insert(std::move(name), std::move(module));
  return std::dynamic_pointer_cast<ModuleType>(base_module);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name, torch::nn::ModuleHolder<ModuleType> module_holder) {
  return register_module(std::move(name), module_holder.ptr());
}
}  // namespace dllm::module

namespace dllm {
void save(const module::Module& module, const std::string& path);

template <typename Module, typename = std::enable_if_t<
                               !std::is_base_of_v<module::Module, Module> &&
                               !std::is_base_of_v<ReadOnlyTensor, Module>>>
static void save(const Module& module, const std::string& path) {
  save(*module, path);
}

void load(const module::Module& module, const std::string& path);

template <typename Module, typename = std::enable_if_t<
                               !std::is_base_of_v<module::Module, Module> &&
                               !std::is_base_of_v<ReadOnlyTensor, Module>>>
static void load(const Module& module, const std::string& path) {
  load(*module, path);
}
}  // namespace dllm