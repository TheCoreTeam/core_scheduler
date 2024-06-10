#include <arpa/inet.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "communication_impl.h"
#include "logger.h"

namespace dllm::communication {
namespace {
template <BackendType backend>
std::unordered_map<MPI_Comm, Comm> &getMap() {
  static std::unordered_map<MPI_Comm, Comm> map;
  return map;
}

Comm createNccl(const MPI_Comm group) {
  int rank, size;
  MPI_Comm_rank(group, &rank);
  MPI_Comm_size(group, &size);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  char addr0[INET_ADDRSTRLEN];  // INET_ADDRSTRLEN is typically 16 for IPv4

  if (rank == 0) {
    const hostent *he = gethostbyname(processor_name);
    DLLM_ASSERT_TRUE(he != nullptr, fmt::format("Error resolving hostname: {}",
                                                hstrerror(h_errno)));
    // Convert the first host address to a string
    strcpy(addr0, inet_ntoa(*reinterpret_cast<in_addr *>(he->h_addr)));
  }

  // Broadcast the IP address from rank 0 to all other ranks
  MPI_Bcast(addr0, INET_ADDRSTRLEN, MPI_CHAR, 0, group);

  auto store = c10::make_intrusive<c10d::TCPStore>(
      addr0, c10d::TCPStoreOptions{.isServer = rank == 0, .numWorkers = size});
  auto backend = c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, size);
  return Comm{std::make_shared<Comm::Impl>(group, std::move(store),
                                           std::move(backend))};
}

Comm lookupMapOrCreate(const MPI_Comm group, const BackendType backendType) {
  static std::mutex mutex;
  std::lock_guard guard{mutex};
  switch (backendType) {
    case NCCL: {
      auto &map = getMap<NCCL>();
      if (const auto find = map.find(group); find == map.end()) {
        auto comm = createNccl(group);
        map.insert({group, comm});
        return comm;
      } else {
        return find->second;
      }
    }
    default: {
      DLLM_ASSERT_TRUE(false, "we only support NCCL now");
    }
  }
}
}  // namespace

const std::shared_ptr<Comm::Impl> &Comm::impl() const { return impl_; }

int64_t Comm::getRank() const { return impl_->backend()->getRank(); }

int64_t Comm::getSize() const { return impl_->backend()->getSize(); }

Comm getComm(const MPI_Comm group, const BackendType backendType) {
  return lookupMapOrCreate(group, backendType);
}

Comm getCommWorld(const BackendType backendType) {
  return getComm(MPI_COMM_WORLD, backendType);
}

Comm getCommNode(const BackendType backendType) {
  static struct MPICommGuard {
    MPI_Comm comm;
    MPICommGuard() {
      int world_rank;
      CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
      CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                    MPI_INFO_NULL, &comm));
    }
    ~MPICommGuard() { CHECK_MPI(MPI_Comm_free(&comm)); }
  } guard;
  return getComm(guard.comm, backendType);
}

Comm::Impl::Impl(const MPI_Comm group, c10::intrusive_ptr<c10d::Store> store,
                 c10::intrusive_ptr<c10d::Backend> backend)
    : group_{group}, store_{std::move(store)}, backend_{std::move(backend)} {}

const c10::intrusive_ptr<c10d::Store> &Comm::Impl::store() const {
  return store_;
}

const c10::intrusive_ptr<c10d::Backend> &Comm::Impl::backend() const {
  return backend_;
}
}  // namespace dllm::communication
