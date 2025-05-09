
## M2N comm library
**Based on these insights, we build our highperformance communication library with the goal of eliminating unnecessary GPU-to-CPU copies, group initialization/handling overhead, and GPU synchronization/memory accesses. Figures 6 and 7 illustrate the sender and receiver architectures and their interactions within our M2N library.**

关键技术：
**GPUDirect** 是一种NVIDIA提供的技术，允许GPU之间直接进行通信，而无需通过CPU内存。
- **GPUDirect RDMA（Remote Direct Memory Access）**：允许一个GPU直接访问另一个GPU的内存，而无需CPU介入。
GPUDirect减少了GPU to CPU的拷贝并在处理小tensor时有显著延迟优势

High-Priotity ACKs：
	we assign ACK packets to high-priority queues, isolating them from data packets, and fine-
	tuning the associated weight configurations empirically.
	将ACK与数据传输隔离，提高ACK的优先级，避免ACK的延迟

