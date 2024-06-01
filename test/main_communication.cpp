#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>
#include <netdb.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "logger.h"
#include "threading/thread_stream_nccl.h"

namespace dllm::test {
ncclUniqueId &getNcclUniqueId() {
  static ncclUniqueId id = [] {
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    return id;
  }();
  return id;
}

auto &getSingleton() {
  struct StreamWrapper {
    MPI_Comm comm;

    ThreadStreamNccl *stream;
    StreamWrapper() {
      int processesPerNode;
      int rank;
      CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                    MPI_INFO_NULL, &comm));
      CHECK_MPI(MPI_Comm_size(comm, &processesPerNode));
      CHECK_MPI(MPI_Comm_rank(comm, &rank));
      stream = new dllm::ThreadStreamNccl{comm, 0};
    }

    ~StreamWrapper() {
      delete stream;
      CHECK_MPI(MPI_Comm_free(&comm));
    }
  };
  static auto stream_wrapper = new StreamWrapper;
  return stream_wrapper;
}

ThreadStreamNccl *getNcclStream() { return getSingleton()->stream; }
}  // namespace dllm::test

namespace {
void test_net() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);  // 正确调用 MPI_Comm_size

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  char addr0[INET_ADDRSTRLEN];  // INET_ADDRSTRLEN is typically 16 for IPv4
                                // addresses

  if (rank == 0) {
    const hostent *he = gethostbyname(processor_name);
    if (he == nullptr) {
      std::cerr << "Error resolving hostname: " << hstrerror(h_errno)
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Convert the first host address to a string
    strcpy(addr0, inet_ntoa(*reinterpret_cast<in_addr *>(he->h_addr)));
    std::cout << "Rank 0 IP Address: " << addr0 << std::endl;
  }

  // Broadcast the IP address from rank 0 to all other ranks
  MPI_Bcast(addr0, INET_ADDRSTRLEN, MPI_CHAR, 0, MPI_COMM_WORLD);

  const c10d::TCPStoreOptions options{.isServer = rank == 0,
                                      .numWorkers = size};
  const auto store = c10::make_intrusive<c10d::TCPStore>(addr0, options);
  c10d::ProcessGroupNCCL group{store, rank, size};
}
}  // namespace

int main(int argc, char **argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  dllm::test::getSingleton();
  ::testing::InitGoogleTest(&argc, argv);

  // test_net();

  const auto error = RUN_ALL_TESTS();
  delete dllm::test::getSingleton();
  dllm::test::getSingleton() = nullptr;
  CHECK_MPI(MPI_Finalize());
  return error;
}
