---
title: DeepEP Tricks
---
## Tricks

## 1. 统计一个bool数组中 1 的出现次数：


- 注重局部性
使用一个warp中的线程进行reduce相加
```C++
__forceinline__ __device__ int warp_reduce_sum(int value) {
	value += __shfl_xor_sync(0xffffffff, value, 16);
	value += __shfl_xor_sync(0xffffffff, value, 8);
	value += __shfl_xor_sync(0xffffffff, value, 4);
	value += __shfl_xor_sync(0xffffffff, value, 2);
	value += __shfl_xor_sync(0xffffffff, value, 1);
	return value;
}
```
每个wrap大小是32，此处每个线程统计从 `start_idx` 到 `end_idx` 之间的数据，每次 idx + 32，
最终计算完成后整个warp进行 `warp_reduce_sum`操作，速度非常快。

线程不是以block为单位调度而是细化到warp为单位


让ChatGPT写一个CUDA统计函数，他这样写：

```Cpp
// CUDA 核函数：统计每个线程块中1的数量
__global__ void countOnesKernel(const int *data, int n, int *blockCounts) {
    __shared__ int sdata[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 每个线程统计自己负责的一个元素
    int val = 0;
    if (idx < n && data[idx] == 1) {
        val = 1;
    }
    sdata[tid] = val;

    __syncthreads();

    // 并行归约，统计当前线程块中1的数量
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 每个线程块的第一个线程写回块内统计结果
    if (tid == 0) {
        blockCounts[blockIdx.x] = sdata[0];
    }
}
```

这段代码输出之后，它这样说：
> 如果你希望我进一步**用 warp-level primitives 优化**，可以把共享内存和同步全部去掉，性能会更高。你是想要我给你写一个 **warp 优化版** 吗？  它会比这个版本快 1.5~2 倍。




## Intra_node的发送接收逻辑

`const bool is_sender = sm_id % 2 == 0;`
将sm_id为偶数的设置为recver，奇数的设置为sender
### sliding window
使用滑动窗口，维护2个变量
`channel_tail_idx`和`channel_head_idx`，表示窗口的头指针和尾指针，
sender在发送之前判断窗口大小，确保 `tail_idx - head_idx < `
recver在copy结束之后递增head_idx


### head tail