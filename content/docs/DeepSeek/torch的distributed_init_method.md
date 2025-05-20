---
draft: true
title: torch的distributed_init_method
---

默认为 `env://`，但是ray中多节点默认是`tcp://{ip}:{port}`,在主节点找到一个可用的端口，占用之后分发给worker节点


多节点的通信基础是ray建立的，
在各个worker中，使用torch.distributed.init(init_method = '') 来初始化分布式环境