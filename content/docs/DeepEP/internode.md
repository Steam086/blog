---
title: internode
---

# DeepEP的inter node设计

inter node部分涉及RDMA，所以总体代码较intra node复杂

### warp_role
inter_node的角色分配
根据sm_id与warp_id确定，在intranode中，只有sender和receiver两种，在inter_node中，出现了更多角色：

根据CUDA线程模型，参考[https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
	每一个kernel执行时的组织从大到小有：grid、block、warp、thread
	这里根据DeepEP的代码，每个block中的线程数目是`(kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32` ，根据代码中的常量计算得到总线程数是512，总warp数是512/32=16
启动一个kernel需要在启动参数中标明block数量和thread数量，这里的block数量是根据任务大小确定的，线程数量是固定的512。

角色分工：
- **RDMASender**(Local copy)
	负责将数据拷贝到RDMA缓冲区（GPU内部拷贝，不涉及通信）
- **NVLReceiver**(Local copy)
	负责接收NVLink数据（节点内通信）
- **RDMAAndNVLForwarder**(NVLink Write)
	Forward tokens from RDMA buffer，从RDMA Buffer中读取数据并写入到NVLink Buffer
- **ForwarderCoordinator**(RDMA Write)
	负责监视RDMAAndNVLForwarder的工作并更新remote head（通知RDMA远端）
- **SenderCoordinator**(Launch async RDMA operation)
	负责启动RDMA数据传输

数据流向:
**RDMASender** -> Forwarder -> **NVLReceiver**
RDMASender把数据拷贝到GPU的RDMA Buffer中，接着SenderCoordinator启动RDMA传输，RDMAAndNVLForwarder读取到RDMA数据之后将数据写入到NVLink Buffer中，NVLReceiver最终从NVL Buffer中读取到数据并拷贝到返回值中

示意图：
![](image/Pasted%20image%2020251014224154.png)

![](image/Pasted%20image%2020251014224229.png)

![](image/Pasted%20image%2020251014224322.png)
**ForwardCoordinator**

这里**ForwardCoordinator**只负责简单的通知远程RDMA（表示Forwarder已经搬运结束，可以继续发送），数据流向与**Sender Coordinator**相反



### 主要函数以及功能

- get_dispatch_layout （纯本地操作）
	*1. 根据topk_ids,统计每个专家的出现次数*
	*2. 统计每个rank上的token数量*
	*3. 统计每个RDMA rank上的token数量*
	*4. 计算is_token_in_rank（二维数组，记录每个每个rank上的）*
- notify_dispatch （通信操作）
	计算出要从每个rank接收的数据总量
- dispatch（通信操作）
	1. 使用滑动窗口，前$\frac{1}{2}$SMs作为`sender`，后$\frac{1}{2}$作为`recver`，将内容传输到缓冲区
	2. 将缓冲区的数据保存到新分配的的`torch::Tensor`
- combine（通信操作）

巧妙设计：
使用bits取代bool变量，
`is_token_in_nvl_rank_bits`相当于一个8bit的bool变量，用于表示该token在nvlink的哪些rank上，发送端将一个token发送到对应的RDMA rank之后，用于确认nvlink上的哪些设备需要此token





