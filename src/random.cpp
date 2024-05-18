#include "random/random.h"

#include "logger.h"
#include "mpi.h"
#include "random/random_internal.h"

namespace dllm::random {
void manual_seed(const uint64_t seed) {
  auto &[_seed, offset] = getRandomState();
  _seed = seed;
  offset = 0;
}

RandomState &getRandomState() {
  static RandomState randomState{
      [] {
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
