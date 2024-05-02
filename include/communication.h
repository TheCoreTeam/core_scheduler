#pragma once

namespace dllm::communication {
enum Backend { MPI, NCCL, NVP2P };

enum Operation { SUM };
}  // namespace dllm::communication
