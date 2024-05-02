#include "communication.h"
#include "tensor.h"

namespace dllm {
template <int N, communication::Backend backend>
struct DTensor;

template <communication::Backend backend>
using DTensor1D = DTensor<1, backend>;
template <communication::Backend backend>
using DTensor2D = DTensor<2, backend>;  // (Row, Col)
template <communication::Backend backend>
using DTensor3D = DTensor<3, backend>;  // (Batch, Sequence, Feature)
template <communication::Backend backend>
using DTensor4D = DTensor<4, backend>;
}  // namespace dllm
