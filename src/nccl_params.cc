#include "nccl_params.h"

#include "debug.h"
#include "param.h"

#define NCCL_PARAM_IMPL(name, env, deftVal)                                    \
  static int64_t cache##name = ncclUninitialized;                              \
  NCCL_PUBLIC int64_t ncclParam##name() {                                      \
    static_assert(deftVal != ncclUninitialized,                                \
                  "default value cannot be the uninitialized value.");         \
    if (__builtin_expect(__atomic_load_n(&cache##name, __ATOMIC_RELAXED) ==    \
                             ncclUninitialized,                                \
                         false)) {                                             \
      ncclLoadParam("NCCL_" env, deftVal, ncclUninitialized, &cache##name);    \
    }                                                                          \
    return cache##name;                                                        \
  }                                                                            \
                                                                               \
  NCCL_PUBLIC void ncclParamReset##name() {                                    \
    if (__atomic_load_n(&cache##name, __ATOMIC_RELAXED) !=                     \
        ncclUninitialized) {                                                   \
      cache##name = ncclUninitialized;                                         \
      INFO(NCCL_ENV, "NCCL_%s cache invalidated. Uninitialized value %lld.",   \
           env, (long long)cache##name);                                       \
    }                                                                          \
  }

#define NCCL_PARAM_INV(name) ncclParamReset##name();

// bootstrap.cc
NCCL_PARAM_IMPL(BootstrapNetEnable, "OOB_NET_ENABLE", 0);
NCCL_PARAM_IMPL(StaggerRate, "UID_STAGGER_RATE", 7000);
NCCL_PARAM_IMPL(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256);

// debug.cc
NCCL_PARAM_IMPL(SetThreadName, "SET_THREAD_NAME", 0);

// enqueue.cc
NCCL_PARAM_IMPL(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);
NCCL_PARAM_IMPL(GraphRegister, "GRAPH_REGISTER", 1);
NCCL_PARAM_IMPL(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);
NCCL_PARAM_IMPL(ChunkSize, "CHUNK_SIZE", 0);
// #if CUDART_VERSION >= 12000
// NCCL uses the "Remote" Mem Sync domain by default
NCCL_PARAM_IMPL(MemSyncDomain, "MEM_SYNC_DOMAIN",
                cudaLaunchMemSyncDomainRemote);
// #endif
NCCL_PARAM_IMPL(NvlsTreeMaxChunkSize, "NVLSTREE_MAX_CHUNKSIZE", -2);

// init.cc
NCCL_PARAM_IMPL(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);
NCCL_PARAM_IMPL(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM_IMPL(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM_IMPL(RuntimeConnect, "RUNTIME_CONNECT", 1);
NCCL_PARAM_IMPL(SetStackSize, "SET_STACK_SIZE", 0);
NCCL_PARAM_IMPL(CGAClusterSize, "CGA_CLUSTER_SIZE", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM_IMPL(MaxCTAs, "MAX_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM_IMPL(MinCTAs, "MIN_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM_IMPL(GdrCopyEnable, "GDRCOPY_ENABLE", 0);
NCCL_PARAM_IMPL(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM_IMPL(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
NCCL_PARAM_IMPL(WorkFifoBytes, "WORK_FIFO_BYTES", NCCL_WORK_FIFO_BYTES_DEFAULT);
NCCL_PARAM_IMPL(WorkArgsBytes, "WORK_ARGS_BYTES", INT64_MAX);
NCCL_PARAM_IMPL(DmaBufEnable, "DMABUF_ENABLE", 1);
NCCL_PARAM_IMPL(MNNVLCliqueId, "MNNVL_CLIQUE_ID", -1);
NCCL_PARAM_IMPL(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM_IMPL(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM_IMPL(Ll128BuffSize, "LL128_BUFFSIZE", -2);
NCCL_PARAM_IMPL(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM_IMPL(P2pPciChunkSize, "P2P_PCI_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM_IMPL(P2pNvlChunkSize, "P2P_NVL_CHUNKSIZE", (1 << 19)); /* 512 kB */
NCCL_PARAM_IMPL(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM_IMPL(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM_IMPL(NvbPreconnect, "NVB_PRECONNECT", 1);
NCCL_PARAM_IMPL(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);
// MNNVL: Flag to indicate whether to enable Multi-Node NVLink
NCCL_PARAM_IMPL(MNNVLEnable, "MNNVL_ENABLE", 2);
NCCL_PARAM_IMPL(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES",
                NCCL_CONFIG_UNDEF_INT);

// proxy.cc
NCCL_PARAM_IMPL(ProxyAppendBatchSize, "PROXY_APPEND_BATCH_SIZE", 16);
NCCL_PARAM_IMPL(CreateThreadContext, "CREATE_THREAD_CONTEXT", 0);
// Set to SIGUSR1 or SIGUSR2 to help debug proxy state during hangs
NCCL_PARAM_IMPL(ProxyDumpSignal, "PROXY_DUMP_SIGNAL", -1);
NCCL_PARAM_IMPL(ProgressAppendOpFreq, "PROGRESS_APPENDOP_FREQ", 8);

// register.cc
NCCL_PARAM_IMPL(LocalRegister, "LOCAL_REGISTER", 1);

// transport.cc
NCCL_PARAM_IMPL(ConnectRoundMaxPeers, "CONNECT_ROUND_MAX_PEERS", 128);
NCCL_PARAM_IMPL(ReportConnectProgress, "REPORT_CONNECT_PROGRESS", 0);

// connect.cc
// Legacy naming
NCCL_PARAM_IMPL(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM_IMPL(MaxNrings, "MAX_NRINGS", -2);
// New naming
NCCL_PARAM_IMPL(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM_IMPL(MaxNchannels, "MAX_NCHANNELS", -2);
NCCL_PARAM_IMPL(UnpackDoubleNChannels, "UNPACK_DOUBLE_NCHANNELS", 1);

// paths.cc

// search.cc
NCCL_PARAM_IMPL(CrossNic, "CROSS_NIC", 2);
// 0: don't use PXN for P2P, 1: use PXN if needed, 2: use PXN as much as
// possible to maximize aggregation
NCCL_PARAM_IMPL(P2pPxnLevel, "P2P_PXN_LEVEL", 2);

// topo.cc
NCCL_PARAM_IMPL(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);
NCCL_PARAM_IMPL(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

// tuning.cc
NCCL_PARAM_IMPL(Nthreads, "NTHREADS", -2);
NCCL_PARAM_IMPL(Ll128Nthreads, "LL128_NTHREADS", -2);
NCCL_PARAM_IMPL(PatEnable, "PAT_ENABLE", 2);
// Network post overhead in ns (1000 = 1 us)
NCCL_PARAM_IMPL(NetOverhead, "NET_OVERHEAD", -2);

// cudawrap.cc
// This env var (NCCL_CUMEM_ENABLE) toggles cuMem API usage
NCCL_PARAM_IMPL(CuMemEnable, "CUMEM_ENABLE", -2);
NCCL_PARAM_IMPL(CuMemHostEnable, "CUMEM_HOST_ENABLE", 0);

// strongstream.cc
NCCL_PARAM_IMPL(GraphMixingSupport, "GRAPH_MIXING_SUPPORT", 1)

// net_ib.cc

// net_socket.cc
NCCL_PARAM_IMPL(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
NCCL_PARAM_IMPL(SocketNthreads, "SOCKET_NTHREADS", -2);

// net.cc
NCCL_PARAM_IMPL(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);
NCCL_PARAM_IMPL(NetSharedComms, "NET_SHARED_COMMS", 1);
// GDRCOPY support: TAIL_ENABLE When enabled locates the RX proxy tail in CUDA
// memory
NCCL_PARAM_IMPL(GdrCopySyncEnable, "GDRCOPY_SYNC_ENABLE", 1);
// GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E read to flush GDRDMA
// buffers
NCCL_PARAM_IMPL(GdrCopyFlushEnable, "GDRCOPY_FLUSH_ENABLE", 0);

// nvls.cc
// #if CUDART_VERSION >= 12010
NCCL_PARAM_IMPL(NvlsEnable, "NVLS_ENABLE", 2);
NCCL_PARAM_IMPL(NvlsChannels, "NVLS_NCHANNELS", 16);
NCCL_PARAM_IMPL(NvlsChunkSize, "NVLS_CHUNKSIZE", 128 * 1024);
// #endif

// p2p.cc
// CE memcpy support
NCCL_PARAM_IMPL(P2pUseCudaMemcpy, "P2P_USE_CUDA_MEMCPY", 0);
// Setting this to non zero causes P2P to use Reads rather than Writes
NCCL_PARAM_IMPL(P2pReadEnable, "P2P_READ_ENABLE", -2);
NCCL_PARAM_IMPL(P2pDirectDisable, "P2P_DIRECT_DISABLE", 0);

// shm.cc
NCCL_PARAM_IMPL(ShmDisable, "SHM_DISABLE", 0);
NCCL_PARAM_IMPL(ShmUseCudaMemcpy, "SHM_USE_CUDA_MEMCPY", 0);
NCCL_PARAM_IMPL(
    ShmMemcpyMode, "SHM_MEMCPY_MODE",
    SHM_SEND_SIDE); // 1 is sender-side, 2 is receiver-side, 3 is both
NCCL_PARAM_IMPL(ShmLocality, "SHM_LOCALITY",
                SHM_RECV_SIDE); // 1 is sender-size, 2 is receiver-size

NCCL_PUBLIC void ncclParamResetAll() {
  // bootstrap.cc
  NCCL_PARAM_INV(BootstrapNetEnable);
  NCCL_PARAM_INV(StaggerRate);
  NCCL_PARAM_INV(StaggerThreshold);

  // debug.cc
  NCCL_PARAM_INV(SetThreadName);

  // enqueue.cc
  NCCL_PARAM_INV(L1SharedMemoryCarveout);
  NCCL_PARAM_INV(GraphRegister);
  NCCL_PARAM_INV(P2pLLThreshold);
  NCCL_PARAM_INV(ChunkSize);
  // #if CUDART_VERSION >= 12000
  // NCCL uses the "Remote" Mem Sync domain by default
  NCCL_PARAM_INV(MemSyncDomain);
  // #endif
  NCCL_PARAM_INV(NvlsTreeMaxChunkSize);

  // init.cc
  NCCL_PARAM_INV(GroupCudaStream);
  NCCL_PARAM_INV(CheckPointers);
  NCCL_PARAM_INV(CommBlocking);
  NCCL_PARAM_INV(RuntimeConnect);
  NCCL_PARAM_INV(SetStackSize);
  NCCL_PARAM_INV(CGAClusterSize);
  NCCL_PARAM_INV(MaxCTAs);
  NCCL_PARAM_INV(MinCTAs);
  NCCL_PARAM_INV(GdrCopyEnable);
  NCCL_PARAM_INV(DisableGraphHelper);
  // GDRCOPY support: FIFO_ENABLE when enabled locates a
  // workFifo in CUDA memory
  NCCL_PARAM_INV(GdrCopyFifoEnable);
  NCCL_PARAM_INV(WorkFifoBytes);
  NCCL_PARAM_INV(WorkArgsBytes);
  NCCL_PARAM_INV(DmaBufEnable);
  NCCL_PARAM_INV(MNNVLCliqueId);
  NCCL_PARAM_INV(BuffSize);
  NCCL_PARAM_INV(LlBuffSize);
  NCCL_PARAM_INV(Ll128BuffSize);
  NCCL_PARAM_INV(P2pNetChunkSize);
  NCCL_PARAM_INV(P2pPciChunkSize);
  NCCL_PARAM_INV(P2pNvlChunkSize);
  NCCL_PARAM_INV(GraphDumpFileRank);
  NCCL_PARAM_INV(CollNetNodeThreshold);
  NCCL_PARAM_INV(NvbPreconnect);
  NCCL_PARAM_INV(AllocP2pNetLLBuffers);
  // MNNVL: Flag to indicate whether to enable Multi-Node NVLink
  NCCL_PARAM_INV(MNNVLEnable);
  NCCL_PARAM_INV(CommSplitShareResources);

  // proxy.cc
  NCCL_PARAM_INV(ProxyAppendBatchSize);
  NCCL_PARAM_INV(CreateThreadContext);
  // Set to SIGUSR1 or SIGUSR2 to help debug proxy state during
  // hangs
  NCCL_PARAM_INV(ProxyDumpSignal);
  NCCL_PARAM_INV(ProgressAppendOpFreq);

  // register.cc
  NCCL_PARAM_INV(LocalRegister);

  // transport.cc
  NCCL_PARAM_INV(ConnectRoundMaxPeers);
  NCCL_PARAM_INV(ReportConnectProgress);

  // connect.cc
  // Legacy naming
  NCCL_PARAM_INV(MinNrings);
  NCCL_PARAM_INV(MaxNrings);
  // New naming
  NCCL_PARAM_INV(MinNchannels);
  NCCL_PARAM_INV(MaxNchannels);
  NCCL_PARAM_INV(UnpackDoubleNChannels);

  // paths.cc

  // search.cc
  NCCL_PARAM_INV(CrossNic);
  // 0: don't use PXN for P2P, 1: use PXN if needed, 2: use PXN
  // as much as possible to maximize aggregation
  NCCL_PARAM_INV(P2pPxnLevel);

  // topo.cc
  NCCL_PARAM_INV(IgnoreCpuAffinity);
  NCCL_PARAM_INV(TopoDumpFileRank);

  // tuning.cc
  NCCL_PARAM_INV(Nthreads);
  NCCL_PARAM_INV(Ll128Nthreads);
  NCCL_PARAM_INV(PatEnable);
  // Network post overhead in ns (1000 = 1 us)
  NCCL_PARAM_INV(NetOverhead);

  // cudawrap.cc
  // This env var (NCCL_CUMEM_ENABLE) toggles cuMem API usage
  NCCL_PARAM_INV(CuMemEnable);
  NCCL_PARAM_INV(CuMemHostEnable);

  // strongstream.cc
  NCCL_PARAM_INV(GraphMixingSupport);

  // net_ib.cc

  // net_socket.cc
  NCCL_PARAM_INV(SocketNsocksPerThread);
  NCCL_PARAM_INV(SocketNthreads);

  // net.cc
  NCCL_PARAM_INV(NetSharedBuffers);
  NCCL_PARAM_INV(NetSharedComms);
  // GDRCOPY support: TAIL_ENABLE When enabled locates the RX
  // proxy tail in CUDA memory
  NCCL_PARAM_INV(GdrCopySyncEnable);
  // GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E
  // read to flush GDRDMA buffers
  NCCL_PARAM_INV(GdrCopyFlushEnable);

  // nvls.cc
  // #if CUDART_VERSION >= 12010
  NCCL_PARAM_INV(NvlsEnable);
  NCCL_PARAM_INV(NvlsChannels);
  NCCL_PARAM_INV(NvlsChunkSize);
  // #endif

  // p2p.cc
  // CE memcpy support
  NCCL_PARAM_INV(P2pUseCudaMemcpy);
  // Setting this to non zero causes P2P to use Reads rather
  // than Writes
  NCCL_PARAM_INV(P2pReadEnable);
  NCCL_PARAM_INV(P2pDirectDisable);

  // shm.cc
  NCCL_PARAM_INV(ShmDisable);
  NCCL_PARAM_INV(ShmUseCudaMemcpy);
  NCCL_PARAM_INV(ShmMemcpyMode);
  NCCL_PARAM_INV(ShmLocality);
}