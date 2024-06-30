#include "module/state.h"

namespace cs::module {
OrderedDict<std::string, Tensor> State::parameters() const {
  return OrderedDict<std::string, Tensor>{};
}

OrderedDict<std::string, State::Increment> State::increments() {
  return OrderedDict<std::string, Increment>{};
}
}  // namespace cs::module
