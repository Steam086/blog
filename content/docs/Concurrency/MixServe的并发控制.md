---
title: MixServe的并发控制
draft: true
---

nccl
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html

在MixServe中，我们想要实现机间通信与机内通信的重叠，形成一个AlltoAll和AllGather的融合通信算子

并发控制：
1. AlltoAll使用pairwise算法，可以在每一轮的pairwise结束之后进行局部的AllGather
2. 设置多个event，在每一轮pairwise结束之后触发这个event，当AllGather检测到这个event发生之后就立即执行。

>[!NOTE]
>这里的AllGather是size不同的AllGather，属于AllGatherV

实现细节：
如何实现AllGatherV size的预分配
可以先预分配一个size，此size根据gather_sizes（没有在TP组上切分前）确定，用于保存AllGatherV的最终结果。

```Cpp
// Fused all2all + broadcast implementation

void fused_all2all_broadcast(

void* send_buffer, size_t send_size,

void** recv_buffers, size_t* recv_sizes,

int num_peers

) {
// Stream 0: Inter-node all2all
// Stream 1: Intra-node broadcast
// Stream 2: Memory operations
// Stream 3: Synchronization
cudaStream_t inter_stream = comm_streams_[0];
cudaStream_t intra_stream = comm_streams_[1];
cudaStream_t mem_stream = comm_streams_[2];

// Step 1: Start inter-node all2all
NCCL_CHECK(ncclGroupStart());

for (int i = 0; i < num_peers; ++i) {
if (i != node_rank_) {
NCCL_CHECK(ncclSend(send_buffer, send_size, ncclInt8,
i, inter_node_comm_, inter_stream));
NCCL_CHECK(ncclRecv(recv_buffers[i], recv_sizes[i], ncclInt8,
i, inter_node_comm_, inter_stream));
}
}

NCCL_CHECK(ncclGroupEnd());
// Record event when inter-node communication starts
CUDA_CHECK(cudaEventRecord(sync_events_[0], inter_stream));
// Step 2: Self-copy while waiting for network
CUDA_CHECK(cudaMemcpyAsync(recv_buffers[node_rank_], send_buffer,
min(send_size, recv_sizes[node_rank_]),
cudaMemcpyDeviceToDevice, mem_stream));

// Step 3: Start intra-node broadcasts as data arrives
CUDA_CHECK(cudaStreamWaitEvent(intra_stream, sync_events_[0], 0));
NCCL_CHECK(ncclGroupStart());

for (int i = 0; i < num_peers; ++i) {
// Broadcast each received buffer within the node
int root = i % 8; // Assuming 8 GPUs per node
NCCL_CHECK(ncclBcast(recv_buffers[i], recv_sizes[i], ncclInt8,
root, intra_node_comm_, intra_stream));
}

NCCL_CHECK(ncclGroupEnd());
// Synchronize all streams
CUDA_CHECK(cudaStreamSynchronize(inter_stream));
CUDA_CHECK(cudaStreamSynchronize(intra_stream));
CUDA_CHECK(cudaStreamSynchronize(mem_stream));

}
```


