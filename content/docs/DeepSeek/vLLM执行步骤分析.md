---
date: '2025-02-11T16:36:38+08:00'
title: 'vLLM执行步骤分析'
math: true
---

以offline为例，整理模型从下载到加载的全流程

### LLM engine

1. init_device
	- 设置torch的device
	`self.device = torch.device(f"cuda:{self.local_rank}")`
	- 初始化分布式环境（最主要），建立tp和pp通信组，确定tp_size和pp_size
2. load_model
	- （可选）如果本地磁盘没有模型，从Hugging Face或者ModelScope下载
	- 从磁盘加载模型，以一个Generator的形式返回（按需加载）
	- 初始化model（`model.__init__`），主要是调用torch.emtpy分配对应dtype的空间
	- load_weights，调用（`model.load_weights`）从磁盘加载的模型权重遍历，如果模型权重名与当前模型的子模块匹配，则将该名称对应的权重通过（`weight_loader`）传入子模块，子模块根据（`tp_size`）和自己的`rank`确定需要加载的模型部分，例如：
	在`ColumnParallelLinear`类的weight_loader中，使用了torch.narrow方法将只取模型权重的一部分copy到实际的param中
	```Java
	loaded_weight = loaded_weight.narrow(output_dim, start_idx,
	                                                 shard_size)
	param_data.copy_(loaded_weight)
	```

1. init_kv_cache
	获取每个worker可用的GPU和CPU blocks
	
