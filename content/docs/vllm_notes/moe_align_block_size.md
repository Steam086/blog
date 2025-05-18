## vLLM中MoE层的`moe_align_block_size`细节介绍

`moe_align_block_size`是Fused_MoE的前置步骤，目的是将经过路由后的token预处理为能进行专家计算的形式

---


目的：得到方便与权重w13矩阵乘的输入张量。模型权重的形状是：`(expert_id, inter_size, hidden_size)`，其中第一维是专家，而且其与输入张量的运算是按照专家划分的，需要将topk_ids转化为一个按照专家id排序的Tensor

###  输入：
`topk_ids`
### 输出：
`sorted_token_ids`按照expert_id排序并padding的topk_ids索引



### 步骤分析

省流版：

1. 计算出每个专家处理的token数，并得到`cumsum`，用于表示该专家之前所有专家的toekn数
2. 遍历`topk_ids`，根据路由到的专家和上一步的`cumsum`，将索引按顺序放进`sorted_token_ids`中。顺便填充`expert_ids`，将专家id放入对应位置
>[!Note]
>步骤2需要一个中间变量记录遍历到当前位置每个专家已经放置的token数，以便确定下一个token应该放置的位置。

详细步骤：

省流版本的分析是针对单线程的，实际计算过程中使用了Triton或者CUDA进行多线程计算，这里以Triton为例

1.  预处理，初始化工具变量`token_cnts`和`cumsum`
```Python
tokens_cnts = torch.zeros((num_experts + 1, num_experts),
	dtype=torch.int32,
	device=topk_ids.device)

cumsum = torch.zeros((num_experts + 1, ),
	dtype=torch.int32,
	device=topk_ids.device)
```
- `cumsum[i]`表示第i个专家之前的token数量
- `tokens_cnts`用来记录每个专家处理的token内部区间内已有多少token（动态变化的）
两者共同确定当前token应该放在sorted_token_ids的哪个位置，因为并发度是`num_experts`,所以`tokens_cnts`的shape是`(num_experts + 1, num_experts)`用来记录多个线程处理的区间内的情况。
2. 四个阶段
- stage 1
计算cumsum
- stage 2
- stage 3
- stage 4

