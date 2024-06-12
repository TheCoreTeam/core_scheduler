#include "random.h"

#include <ATen/Context.h>
#include <mpi.h>

#include "logger.h"

namespace dllm::random {
RandomState &getRandomState();

void manual_seed(const uint64_t seed) {
  auto &[_seed, offset] = getRandomState();
  _seed = seed;
  offset = 0;
  at::manual_seed(seed);
}

RandomState &getRandomState() {
  static RandomState randomState{
      []() {
        int init = false;
        CHECK_MPI(MPI_Initialized(&init));
        int worldRank = 0;
        if (init) {
          CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
        }
        return static_cast<uint64_t>(worldRank);
      }(),
      0};
  return randomState;
}
}  // namespace dllm::random
