#include "module/module.h"

#include <fmt/format.h>

namespace dllm::module {
void Module::apply_to_submodules(
    const NamedModulePointerApplyFunction& function,
    const std::string& name_prefix) const {
  for (const auto& child : children_) {
    auto qualified_name = fmt::format("{}.{}", name_prefix, child.key());
    function(qualified_name, child.value());
    child.value()->apply_to_submodules(function, qualified_name);
  }
}

std::shared_ptr<Module> Module::shared_from_this_checked() const {
  std::shared_ptr<const Module> ptr;
  try {
    ptr = shared_from_this();
  } catch (const std::bad_weak_ptr&) {
    DLLM_ASSERT_TRUE(
        false,
        "It looks like you attempted to retrieve your top-level module "
        "as a shared_ptr, but it is not stored in a shared_ptr. "
        "Use std::make_shared instead of creating your module on "
        "the stack, or alternatively do not try to access your top-level "
        "module at all by passing /*include_self=*/false "
        "to modules() or named_modules()");
  }
  return std::const_pointer_cast<Module>(ptr);
}

void Module::apply(const ConstNamedModuleApplyFunction& function,
                   const std::string& name_prefix) const {
  function(/*name=*/name_prefix, *this);
  apply_to_submodules(
      [&function](const std::string& name,
                  const std::shared_ptr<Module>& module) {
        function(name, *module);
      },
      name_prefix);
}

OrderedDict<std::string, std::shared_ptr<State>> Module::named_states(
    const bool recurse) const {
  OrderedDict<std::string, std::shared_ptr<State>> result;
  if (!recurse) {
    for (const auto& state : states_) {
      result.insert(state.key(), state.value());
    }
  } else {
    apply([&result](const std::string& name, const Module& module) {
      for (const auto& state : module.named_states(/*recurse=*/false)) {
        result.insert(fmt::format("{}.{}", name, state.key()), state.value());
      }
    });
  }
  return result;
}

void Module::register_state(std::string name, std::shared_ptr<State> state) {
  states_.insert(std::move(name), std::move(state));
}
}  // namespace dllm::module
