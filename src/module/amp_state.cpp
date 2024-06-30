#include "module/amp_state.h"

namespace cs::module {
OrderedDict<std::string, Tensor> AmpState::parametersFp32() const {
  return OrderedDict<std::string, Tensor>{};
}

OrderedDict<std::string, Tensor> AmpState::parameters() const {
  return parametersFp32();
}
}  // namespace cs::module
