---
title: 关机很慢
date: 2025-02-25T13:06:37+08:00
draft: false
math: true
---
### 遇到的问题：

1. （已解决）源码下载之后依赖安装遇到问题，先解决环境配置问题
2. （不必理会）model的MoE模块中有关于量化的内容不了解
3. （已解决）原有代码对Pytorch的profile做了调整，还能否正确得到profile数据？
4. 代码中实现了MLA，如何对代码中的MoE进行替换，替换之后weight_loader要如何重写？


## weight_loader是如何实现的



初始化pp_groups的时候，会有一个`start_layer, end_layer` 用于加载指定区间的模型参数

PPMissingLayer()一个占位符 `PPMissingLayer(torch.nn.Identity):`

- 模型中每个层的参数都有一个 `prefix: str`，应该是用于给每个参数一个name，用于加载对应的模型weights。比如：第101层DecoderLayer的SelfAttention中将hidden_states转化为Query的参数可以表示为`"model.layers.101.self_attn.q_proj"`，
- 要实现张量并行，需要将此参数加载时分配到不同的GPU中，假设此参数的shape为(800, 800)，tp_size为8，则需要将参数矩阵竖切，分配到每个设备上，每个设备上的此参数的shape均为(800, 100)

> Note
> 实际加载过程中，还要考虑多头注意力的情况，如head数量为100，但是tp_size=7，此时无法将head在各个设备上均匀分配，会出现报错。


### 具体实现：

一个executor控制多个worker，一个worker对应一个设备
#### worker的工作流程（加载、执行）：
- 创建worker
- 初始化设备init_device初始化分布式环境init_distributed_env (与其他线程建立通信)
- 加载模型load_model(从磁盘到内存)
- load_weights（需要上一步的tp_rank和pp_rank确定需要加载哪部分模型）
每个的layer都需要自定义的weights_loader，
在模型层面，一个worker上的模型在执行初始化方法时会根据自身处于pipeline中的位置将self.paramList初始化为一个列表，假设某个worker中的模型处于流水线的第3个阶段，则其模型会被初始化为`[placeholder,placeholder,real_param,placeholder]`，placeholder中没有权重，不会被加载，实际加载过程只加载real_param中的权重。

load_model步骤拆解
- 确定模型架构
- 根据模型架构确定model对应的class
- 调用模型初始化代码（torch.empty）预分配对应dtype的空间
- 加载模型参数download并load_weights

Question：
怎么把流水线并行他需要的模型的参数传给loader？？

回答： 不做传递，判断是不是placeholder（PPMissingLayer），如果是，就不加载
#### 执行阶段：

TODO 明天重点看worker和pipeline


