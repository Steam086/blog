---
title: DeepEp
---
DeepEP对于任务的分配

### SM与warp的任务分配
- 所有偶数SMs负责send，奇数SMs负责recv
- 多个warp负责一个rank

```c++
// Several warps are response for a single rank
const auto num_threads_per_rank = kNumThreads / kNumRanks;
```


#### sender根据sm_id获取token范围
```cpp
__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx, int& token_end_idx) {
	int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
	token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
	token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}
```
#### recver根据什么获取任务范围？
recver不获取任务范围，直接从缓冲区读取数据并放到相应的位置
因为每一个warp只负责一个rank的数据，所以copy的数据是相对比较连续的
### buffer的布局：
有`buffer_ptr[num_ranks]: void*`，
对`buffer_ptr[i]`来说，其布局为
- rank_prefix_matrix
- channel_start_offset
- channel_end_offset
- channel_head_idx
- channel_tail_idx
上述每一个`channel_xxx_idx(offset) `都是一个长度为`num_channels_total`的buffer,
指向具体的rank的buffer，根据偏移量计算得出

每一个Buffer的大小是：
~~notify_dispatch计算得出的`num_recv_buffer_tokens`~~
预先定义的Config `num_max_nvl_chunked_recv_tokens;`
总的buffer大小是实例化Buffer对象时候传入的，在测试用例中，Buffer大小是2e9

### receiver在while轮询滑动窗口的tail， sender在发送完成之后更新tail

接收者在轮询检测到有数据发送时，分别计算出接收的数据在最终recv_token和buffer中的位置，数据从buffer-> recv_token

>[!Note]
>接收端并不根据自身所处的channel分配任务，而是直接从Buffer中读取出实际数据和数据对应的**SourceMeta**，来判断这个数据来自哪个rank



