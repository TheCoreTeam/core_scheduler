#pragma once
#include <c10/util/Exception.h>
#include <torch/nn/pimpl.h>
#include <torch/ordered_dict.h>

#include <memory>
#include <string>

#include "logger.h"
#include "module/state.h"
#include "tensor.h"

namespace dllm::module {
struct Module : std::enable_shared_from_this<Module> {
  virtual ~Module() = default;
  // using ModuleApplyFunction = std::function<void(Module&)>;
  // using ConstModuleApplyFunction = std::function<void(const Module&)>;
  using NamedModulePointerApplyFunction =
      std::function<void(const std::string&, const std::shared_ptr<Module>&)>;
  using ConstNamedModuleApplyFunction =
      std::function<void(const std::string&, const Module&)>;
  // using NamedModuleApplyFunction =
  //     std::function<void(const std::string&, Module&)>;

  void apply_to_submodules(
      const NamedModulePointerApplyFunction& function,
      const std::string& name_prefix = std::string()) const;

  std::shared_ptr<Module> shared_from_this_checked() const;

  // void apply(const ModuleApplyFunction& function) {
  //   function(*this);
  //   apply_to_submodules(
  //       [&function](const std::string&, const std::shared_ptr<Module>&
  //       module) {
  //         function(*module);
  //       });
  // }

  // void apply(const ConstModuleApplyFunction& function) const {
  //   function(*this);
  //   apply_to_submodules(
  //       [&function](const std::string&, const std::shared_ptr<Module>&
  //       module) {
  //         function(*module);
  //       });
  // }

  // void apply(const NamedModulePointerApplyFunction& function,
  //            const std::string& name_prefix = {}) const {
  //   function(
  //       /*name=*/name_prefix, shared_from_this_checked());
  //   apply_to_submodules(function, name_prefix);
  // }

  void apply(const ConstNamedModuleApplyFunction& function,
             const std::string& name_prefix = {}) const;

  // void apply(const NamedModuleApplyFunction& function,
  //            const std::string& name_prefix = {}) {
  //   function(/*name=*/name_prefix, *this);
  //   apply_to_submodules(
  //       [&function](const std::string& name,
  //                   const std::shared_ptr<Module>& module) {
  //         function(name, *module);
  //       },
  //       name_prefix);
  // }

  void register_state(std::string name, std::shared_ptr<State> state);

  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name, std::shared_ptr<ModuleType> module);

  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name, torch::nn::ModuleHolder<ModuleType> module_holder);

  OrderedDict<std::string, std::shared_ptr<State>> named_states(
      bool recurse = true) const;

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
