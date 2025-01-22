#ifndef NCCL_PARAMS_H_
#define NCCL_PARAMS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ncclUninitialized INT64_MIN

#ifndef NCCL_PUBLIC
#define NCCL_PUBLIC __attribute__((visibility("default")))
#endif

#define NCCL_PARAM_DECL(name, env, deftVal)                                    \
  NCCL_PUBLIC int64_t ncclParam##name();                                       \
  NCCL_PUBLIC void ncclParamReset##name();

// bootstrap.cc
NCCL_PARAM_DECL(BootstrapNetEnable, "OOB_NET_ENABLE", 0);
NCCL_PARAM_DECL(StaggerRate, "UID_STAGGER_RATE", 7000);
NCCL_PARAM_DECL(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256);
NCCL_PARAM_DECL(RasEnable, "RAS_ENABLE", 1);

// debug.cc
NCCL_PARAM_DECL(SetThreadName, "SET_THREAD_NAME", 0);

// enqueue.cc
NCCL_PARAM_DECL(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);
NCCL_PARAM_DECL(GraphRegister, "GRAPH_REGISTER", 1);
NCCL_PARAM_DECL(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);
NCCL_PARAM_DECL(ChunkSize, "CHUNK_SIZE", 0);
// #if CUDART_VERSION >= 12000
// NCCL uses the "Remote" Mem Sync domain by default
NCCL_PARAM_DECL(MemSyncDomain, "MEM_SYNC_DOMAIN",
                cudaLaunchMemSyncDomainRemote);
// #endif
NCCL_PARAM_DECL(NvlsTreeMaxChunkSize, "NVLSTREE_MAX_CHUNKSIZE", -2);

// init.cc
#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM                                                 \
  0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM                                                 \
  1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

#define NCCL_CONFIG_UNDEF_INT INT_MIN

NCCL_PARAM_DECL(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);
NCCL_PARAM_DECL(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM_DECL(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM_DECL(RuntimeConnect, "RUNTIME_CONNECT", 1);
NCCL_PARAM_DECL(SetStackSize, "SET_STACK_SIZE", 0);
NCCL_PARAM_DECL(CGAClusterSize, "CGA_CLUSTER_SIZE", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM_DECL(MaxCTAs, "MAX_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM_DECL(MinCTAs, "MIN_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM_DECL(GdrCopyEnable, "GDRCOPY_ENABLE", 0);
NCCL_PARAM_DECL(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM_DECL(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
#define NCCL_WORK_FIFO_BYTES_DEFAULT (1 << 20)
NCCL_PARAM_DECL(WorkFifoBytes, "WORK_FIFO_BYTES", NCCL_WORK_FIFO_BYTES_DEFAULT);
NCCL_PARAM_DECL(WorkArgsBytes, "WORK_ARGS_BYTES", INT64_MAX);
NCCL_PARAM_DECL(DmaBufEnable, "DMABUF_ENABLE", 1);
NCCL_PARAM_DECL(MNNVLCliqueId, "MNNVL_CLIQUE_ID", -1);
#define DEFAULT_LL_BUFFSIZE                                                    \
  (NCCL_LL_LINES_PER_THREAD * NCCL_LL_MAX_NTHREADS * NCCL_STEPS *              \
   sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE                                                 \
  (NCCL_LL128_ELEMS_PER_THREAD * NCCL_LL128_MAX_NTHREADS * NCCL_STEPS *        \
   sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
NCCL_PARAM_DECL(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM_DECL(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM_DECL(Ll128BuffSize, "LL128_BUFFSIZE", -2);
NCCL_PARAM_DECL(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM_DECL(P2pPciChunkSize, "P2P_PCI_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM_DECL(P2pNvlChunkSize, "P2P_NVL_CHUNKSIZE", (1 << 19)); /* 512 kB */
NCCL_PARAM_DECL(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM_DECL(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM_DECL(NvbPreconnect, "NVB_PRECONNECT", 1);
NCCL_PARAM_DECL(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);
// MNNVL: Flag to indicate whether to enable Multi-Node NVLink
NCCL_PARAM_DECL(MNNVLEnable, "MNNVL_ENABLE", 2);
NCCL_PARAM_DECL(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES",
                NCCL_CONFIG_UNDEF_INT);

// proxy.cc
NCCL_PARAM_DECL(ProxyAppendBatchSize, "PROXY_APPEND_BATCH_SIZE", 16);
NCCL_PARAM_DECL(CreateThreadContext, "CREATE_THREAD_CONTEXT", 0);
// Set to SIGUSR1 or SIGUSR2 to help debug proxy state during hangs
NCCL_PARAM_DECL(ProxyDumpSignal, "PROXY_DUMP_SIGNAL", -1);
NCCL_PARAM_DECL(ProgressAppendOpFreq, "PROGRESS_APPENDOP_FREQ", 8);

// register.cc
NCCL_PARAM_DECL(LocalRegister, "LOCAL_REGISTER", 1);

// transport.cc
NCCL_PARAM_DECL(ConnectRoundMaxPeers, "CONNECT_ROUND_MAX_PEERS", 128);
NCCL_PARAM_DECL(ReportConnectProgress, "REPORT_CONNECT_PROGRESS", 0);

// connect.cc
// Legacy naming
NCCL_PARAM_DECL(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM_DECL(MaxNrings, "MAX_NRINGS", -2);
// New naming
NCCL_PARAM_DECL(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM_DECL(MaxNchannels, "MAX_NCHANNELS", -2);
NCCL_PARAM_DECL(UnpackDoubleNChannels, "UNPACK_DOUBLE_NCHANNELS", 1);

// paths.cc
NCCL_PARAM_DECL(NvbDisable, "NVB_DISABLE", 0);
NCCL_PARAM_DECL(IgnoreDisabledP2p, "IGNORE_DISABLED_P2P", 0);
NCCL_PARAM_DECL(NetGdrRead, "NET_GDR_READ", -2);
// Set to 0 to disable the flush on Hopper when using GDR
NCCL_PARAM_DECL(NetForceFlush, "NET_FORCE_FLUSH", 0);
NCCL_PARAM_DECL(NetDisableIntra, "NET_DISABLE_INTRA", 0);
NCCL_PARAM_DECL(PxnDisable, "PXN_DISABLE", 0);
NCCL_PARAM_DECL(NChannelsPerNetPeer, "NCHANNELS_PER_NET_PEER", -1);
NCCL_PARAM_DECL(MinP2pNChannels, "MIN_P2P_NCHANNELS", 1);
#ifndef MAXCHANNELS
#define MAXCHANNELS 32
#endif
NCCL_PARAM_DECL(MaxP2pNChannels, "MAX_P2P_NCHANNELS", MAXCHANNELS);

// search.cc
NCCL_PARAM_DECL(CrossNic, "CROSS_NIC", 2);
// 0: don't use PXN for P2P, 1: use PXN if needed, 2: use PXN as much as
// possible to maximize aggregation
NCCL_PARAM_DECL(P2pPxnLevel, "P2P_PXN_LEVEL", 2);

// topo.cc
NCCL_PARAM_DECL(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);
NCCL_PARAM_DECL(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

// tuning.cc
NCCL_PARAM_DECL(Nthreads, "NTHREADS", -2);
NCCL_PARAM_DECL(Ll128Nthreads, "LL128_NTHREADS", -2);
NCCL_PARAM_DECL(PatEnable, "PAT_ENABLE", 2);
// Network post overhead in ns (1000 = 1 us)
NCCL_PARAM_DECL(NetOverhead, "NET_OVERHEAD", -2);

// cudawrap.cc
// This env var (NCCL_CUMEM_ENABLE) toggles cuMem API usage
NCCL_PARAM_DECL(CuMemEnable, "CUMEM_ENABLE", -2);
NCCL_PARAM_DECL(CuMemHostEnable, "CUMEM_HOST_ENABLE", -1);

// strongstream.cc
NCCL_PARAM_DECL(GraphMixingSupport, "GRAPH_MIXING_SUPPORT", 1)

// net_ib.cc
NCCL_PARAM_DECL(IbGidIndex, "IB_GID_INDEX", -1);
NCCL_PARAM_DECL(IbRoutableFlidIbGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);
NCCL_PARAM_DECL(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
NCCL_PARAM_DECL(IbTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM_DECL(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM_DECL(IbPkey, "IB_PKEY", 0);
NCCL_PARAM_DECL(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM_DECL(IbSl, "IB_SL", 0);
NCCL_PARAM_DECL(IbTc, "IB_TC", 0);
NCCL_PARAM_DECL(IbArThreshold, "IB_AR_THRESHOLD", 8192);
NCCL_PARAM_DECL(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM_DECL(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM_DECL(IbFifoTc, "IB_FIFO_TC", -1);
NCCL_PARAM_DECL(IbAsyncEvents, "IB_RETURN_ASYNC_EVENTS", 1);
NCCL_PARAM_DECL(IbEceEnable, "IB_ECE_ENABLE", 1);
NCCL_PARAM_DECL(IbDisable, "IB_DISABLE", 0);
NCCL_PARAM_DECL(IbMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM_DECL(IbMergeNics, "IB_MERGE_NICS", 1);
NCCL_PARAM_DECL(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);
NCCL_PARAM_DECL(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);
NCCL_PARAM_DECL(IbWarnRailLocal, "IB_WARN_RAIL_LOCAL", 0);
NCCL_PARAM_DECL(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);

// ibwrap.cc
NCCL_PARAM_DECL(IbMQpRetryAll, "IB_MQP_RETRY_ALL", 0);
NCCL_PARAM_DECL(IbMQpRetryCnt, "IB_MQP_RETRY_CNT", 34);
NCCL_PARAM_DECL(IbMQpRetryTimeout, "IB_MQP_RETRY_SLEEP_MSEC", 100); // in milliseconds

// socket.cc
NCCL_PARAM_DECL(RetryCnt, "SOCKET_RETRY_CNT", 34);
NCCL_PARAM_DECL(RetryTimeOut, "SOCKET_RETRY_SLEEP_MSEC", 100);

// net_socket.cc
NCCL_PARAM_DECL(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
NCCL_PARAM_DECL(SocketNthreads, "SOCKET_NTHREADS", -2);

// net.cc
NCCL_PARAM_DECL(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);
NCCL_PARAM_DECL(NetSharedComms, "NET_SHARED_COMMS", 1);
// GDRCOPY support: TAIL_ENABLE When enabled locates the RX proxy tail in CUDA
// memory
NCCL_PARAM_DECL(GdrCopySyncEnable, "GDRCOPY_SYNC_ENABLE", 1);
// GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E read to flush GDRDMA
// buffers
NCCL_PARAM_DECL(GdrCopyFlushEnable, "GDRCOPY_FLUSH_ENABLE", 0);
NCCL_PARAM_DECL(NetOptionalRecvCompletion, "NET_OPTIONAL_RECV_COMPLETION", 1);

// nvls.cc
// #if CUDART_VERSION >= 12010
NCCL_PARAM_DECL(NvlsEnable, "NVLS_ENABLE", 2);
NCCL_PARAM_DECL(NvlsChannels, "NVLS_NCHANNELS", 16);
NCCL_PARAM_DECL(NvlsChunkSize, "NVLS_CHUNKSIZE", 128 * 1024);
// #endif

// p2p.cc
// CE memcpy support
NCCL_PARAM_DECL(P2pUseCudaMemcpy, "P2P_USE_CUDA_MEMCPY", 0);
// Setting this to non zero causes P2P to use Reads rather than Writes
NCCL_PARAM_DECL(P2pReadEnable, "P2P_READ_ENABLE", -2);
NCCL_PARAM_DECL(P2pDirectDisable, "P2P_DIRECT_DISABLE", 0);
NCCL_PARAM_DECL(LegacyCudaRegister, "LEGACY_CUDA_REGISTER", 0);

// shm.cc
#define SHM_SEND_SIDE 1
#define SHM_RECV_SIDE 2
NCCL_PARAM_DECL(ShmDisable, "SHM_DISABLE", 0);
NCCL_PARAM_DECL(ShmUseCudaMemcpy, "SHM_USE_CUDA_MEMCPY", 0);
// 1 is sender-side, 2 is receiver-side, 3 is both
NCCL_PARAM_DECL(ShmMemcpyMode, "SHM_MEMCPY_MODE", SHM_SEND_SIDE);
// 1 is sender-size, 2 is receiver-size
NCCL_PARAM_DECL(ShmLocality, "SHM_LOCALITY", SHM_RECV_SIDE);

// ras.cc
NCCL_PARAM_DECL(RasTimeoutFactor, "RAS_TIMEOUT_FACTOR", 1);

NCCL_PUBLIC void ncclParamResetAll();

#ifdef __cplusplus
}
#endif

#endif