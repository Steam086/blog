---
title: FusedMoE
slug: fusedmoe-oy12q
url: /post/fusedmoe-oy12q.html
date: '2025-05-09 16:33:11+08:00'
lastmod: '2025-05-09 17:30:00+08:00'
toc: true
isCJKLanguage: true
---



# FusedMoE

### 阶段 1： DP 广播

如果有 dp，则在 DP 组内广播，此时所有的计算节点拥有相同的 token

### 阶段 2:   moe_align_block_size(padding)

输入：`topk_ids, expert_map=None`​

在需要 EP 时传入 `expert_map`​，将本地专家 id 映射到本地专家 id

输出：

* ​`num_tokens_post_padded`​

  padding 操作之后的 token 总数（完全不 padding 的情况下这个值为 `topk * num_tokens`​）,padding 之后保证每个专家处理的 token 数都能被 block_size 整除
* ​`sorted_token_ids`​,
  shape: `(num_post_padding, )`​
  按 `expert_id` ​排序的 token 索引。

  实际保存的值是 token 在 `topk_ids` ​中的偏移量，这样一来，对于 sorted_token_ids 中的每一个值，都有：

  $$
  token\_id = sorted\_token\_ids\  //\ topk
  $$

  既确定了该位置的 token_id，也能确定该 token 的 `topk_weights`​

  下图是 `expert_size=4, block_size=3` ​的案例，此处的 `num_tokens_post_padded`​=18

  ![Pasted image 20250509141640](/content/docs/images/Pasted image 20250509141640-20250510180232-46xf8ae.png)​

‍

* ​`expert_ids`​
  shape:  `(num_post_padding, )`​
  ​`sorted_token_ids` ​每个位置对应的专家 id，用于在 kernel 中确定待处理的 token 是否属于 local expert，如果不属于，在计算时则向输出矩阵中写入 0,代码如下所示

```Python
off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
if off_experts == -1:
	# -----------------------------------------------------------
	# Write back zeros to the output when the expert is not
	# in the current expert parallel rank.
	write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
	offs_token, token_mask, BLOCK_SIZE_M,
	BLOCK_SIZE_N, compute_type)
	
	return
```

> Note
>
> 这里步骤 1 和 2 的输出矩阵初始化时都没有使用 torch.zero 而是使用了 torch.empty，所以不能默认没有写入的位置是 0，所以才有上面那段代码的写入 0 的操作

### 阶段 3: GEMM 计算

1. 在 Triton kernel 中遍历 `sorted_token_ids`​，根据索引加载输入并与 `w13` ​权重进行计算
2. 将步骤 `1.` ​的输出与权重 `w2` ​的矩阵运算得到最终的结果，最后计算与 `topk_weights` ​的乘积
3. ​`moe_sum`​，将计算结果转化为原始的 token
   这一步的流程是：可以简单理解为对上一步的输出（shape=(num_tokens, topk, hidden_size)）进行一个 `torch.sum(input, dim=1)` ​的操作

### 阶段 4: allreduce

先进行 DP 组内的 allreduce，只保留本地 token 的结果

> Note
>
> 这里的allreduce操作确实有冗余，但是涉及数据并行的情况一般都数据规模较大，某一个dp_rank节点上的token分散到各个节点上的概率更大，这时通信冗余不在明显

```python
        if self.dp_size > 1:
            start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
                self.dp_rank - 1]
            end = cu_tokens_across_dp_cpu[self.dp_rank]

            all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
            final_hidden_states = all_hidden_states[start:end, :]
```

再进行 TP 组内的 allreduce

‍
