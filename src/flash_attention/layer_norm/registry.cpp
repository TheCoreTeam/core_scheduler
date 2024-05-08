#include "ln.h"

namespace layer_norm {
FwdRegistry FWD_FUNCS, PARALLEL_FWD_FUNCS;
BwdRegistry BWD_FUNCS, PARALLEL_BWD_FUNCS;
}  // namespace layer_norm
