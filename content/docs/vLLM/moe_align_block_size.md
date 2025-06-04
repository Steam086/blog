---
type: docs
title: moe_align_block_size
math: true
---


## vLLM中MoE层的`moe_align_block_size`细节介绍

`moe_align_block_size`是[FusedMoE](../fusedmoe)的前置步骤，目的是将经过路由后的token预处理为能方便与权重计算的形式

---


目的：得到方便与权重w13矩阵乘的输入张量。模型权重的形状是：`(expert_id, inter_size, hidden_size)`，其中第一维是专家，而且其与输入张量的运算是按照专家划分的，需要将topk_ids转化为一个按照专家id排序的Tensor

**输入：**
`topk_ids` 

经过moe路由后的结果，`topk_ids`表示每个token对应的专家id，`topk_weights`表示`topk_ids`对应的专家的路由权重，两者shape一致。

**输出：**
`sorted_token_ids`按照专家id排序并padding的`topk_ids`索引


---
### 步骤分析

### 省流版：

1. 计算出每个专家处理的token数，并得到`cumsum`，用于表示该专家之前所有专家的toekn数
2. 遍历`topk_ids`，根据路由到的专家和上一步的`cumsum`，将索引按顺序放进`sorted_token_ids`中。顺便填充`expert_ids`，将专家id放入对应位置
>[!Note]
>步骤2需要一个中间变量记录遍历到当前位置每个专家已经放置的token数，以便确定下一个token应该放置的位置。
步骤2的单线程代码：
```Python
topk_ids = ? # 传入的参数shape = (num_tokens, topk)
sorted_token_ids = ? # 一个空的，用 num_experts 填充的张量，shape = (num_tokens_post_padded)
cumsum = ? # 表示该专家处理的第一个token的位置, shape = (num_experts)

def generate_sorted_id(topk_ids: torch.Tensor, cumsum: torch.Tensor, sorted_token_ids: torch.Tensor)
	for i, expert_id in enumerate(topk_ids):
		sorted_token_ids[cumsum[expert_id]] = i
		cumsum[expert_id] += 1
```
简单将其扩展为多线程（在GPU上执行的）Triton代码即为真实代码逻辑
>[!important]
>扩展为多线程代码需要增加一些类似`cumsum`的counter变量来确定当前线程的每个expert总计有多少token位置已经被占用，从而确定`topk_ids`应该放在`sorted_token_ids`中的位置

比如源代码中的逻辑使用了一个`tokens_cnts = torch.zeros((num_experts + 1, num_experts),)`来为每个线程计数。每个线程处理topk_ids中的一部分（除了最后一个线程外都是`numel // num_experts`）。最终启动的Block数是`num_experts`，`token_cnt`的前`num_experts`行就是针对每个线程的计数，最后一行表示`cumsum`


---
#### 详细步骤：

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
- `cumsum[i]`表示第i个专家之前的token数量，包括padding
- `tokens_cnts`用来记录每个专家处理的token内部区间内已有多少token（动态变化的）
两者共同确定当前token应该放在sorted_token_ids的哪个位置，因为并发度是`num_experts`,所以`tokens_cnts`的shape是`(num_experts + 1, num_experts)`用来记录多个线程处理的区间内的情况。
2. 4个阶段
其中只有stage 3 为单线程，其余线程数均为`num_experts`
- **stage 1**  （横向计算）
	多线程计算每个部分的cumsum
- **stage 2**  （竖向相加）
	将上一步计算的每一个部分加起来
加起来
- **stage 3**  单线程（考虑padding计算出`cumsum`和`num_tokens_post_pad`）
	计算`num_tokens_post_pad`，计算包括padding的`cumsum`
- **stage 4**
	根据前面三个阶段的计算结果生成`sorted_token_ids`，这一步需要使用到`tokens_cnts`和`cumsum`来确定每个token在sorted_token_ids中的索引

