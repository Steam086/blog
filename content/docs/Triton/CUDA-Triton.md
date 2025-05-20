---
title: CUDA Triton
date: 2025-05-12T15:56:45+08:00
math: true
type: blog
---

## CUDA 与 Triton 概述

本概述旨在深入探讨 NVIDIA CUDA 编程模型及其内存层次结构、调度机制和同步方法，并进一步介绍 CUDA Stream 和 CUDA Graph 等高级优化技术。最后，我们将剖析 Triton 这一新兴的 GPU 编程语言，对比其与 CUDA 的异同，并揭示其在深度学习领域性能优化的独特优势。



## CUDA


**CUDA** 是 NVIDIA 推出的一种并行计算平台和编程模型，它允许开发者利用 GPU 的并行处理能力来加速计算密集型任务。
### CUDA编程模型：

在 CUDA 编程中，计算任务被组织成一个层次结构：

- **Grid**
- **Block**
- **Thread**
Grid由多个Block组成，而Block由多个线程组成
![](image/Pasted%20image%2020250514163426.png)



### Memory Hierarchy

- **Registers**
速度最快，在CUDA kernel中声明的变量默认分配到寄存器中。

如果使用超出寄存器容量限制的变量，编译器会将一部分溢出的变量放入Local Memory。单个线程最大的寄存器使用量一般为255
- **Shared Memory**
属于 SM，供 **同一个 Block 的线程共享**，访问速度接近寄存器，远快于全局内存，大小一般为 **48KB 或 96KB**，与 L1 Cache 通常**共享物理空间**，在CUDA编程中共享内存通过`__shared__`声明
- **L1 Cache**
L1 Cache与Shared Memory共享硬件资源
- **L2 Cache**
所有SMs共享，处在HBM和SMs之间
>Starting with CUDA 11.0, devices of compute capability 8.0 and above have the capability to influence persistence of data in the L2 cache, potentially providing higher bandwidth and lower latency accesses to global memory.


可以将L2 Cache中的一部分预留出来用于持久化访问全局内存中的数据
```C++
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/
```
- **Local Memory**
地位与全局内存相同，也会在L1 L2 Cache中缓存，在单个Thread的寄存器不足以保存变量时会发生spilling，将寄存器无法保存的变量存到 Local Memory 中
>Local memory is cached in an L1 then an L2 cache so register spilling may not mean a dramatic performance decrease.

- **Global Memory**
通常说的显存或者HBM，速度最慢，容量最大，可以通过`cudaMalloc`, `cudaFree`, `cudaMemcpy` `cudaMemset` 等API操控。
![](image/Pasted%20image%2020250521010931.png)

| GPU型号            | 架构           | 显存容量  | 显存带宽 (GB/s) |
| ---------------- | ------------ | ----- | ----------- |
| **H200**         | Hopper       | 141GB | ~4,800GB/s  |
| **H100 SXM**     | Hopper       | 80 GB | ~3,350 GB/s |
| **H100 PCIe**    | Hopper       | 80 GB | ~2,000 GB/s |
| **A100 80GB**    | Ampere       | 80 GB | ~2,039 GB/s |
| **A100 40GB**    | Ampere       | 40 GB | ~1,555 GB/s |
| **V100 32GB**    | Volta        | 32 GB | ~900 GB/s   |
| **V100 16GB**    | Volta        | 16 GB | ~900 GB/s   |
| **RTX 6000 Ada** | Ada Lovelace | 48 GB | ~960 GB/s   |


### 调度

#### SM
一个Block是分配在一个SM（Stream Multiprocessor）上执行的
- 一个 SM 可以同时运行多个 Block（具体数量取决于资源：register 数、shared memory、warp 数量等）
- 一个 Block 不会被拆分到多个 SM 上

#### warp
- 一个 warp（线程束）就是 GPU 中调度和执行的基本单元，**固定为 32 个线程**。这 32 个线程**同时执行相同的指令**（SIMT 模型）。
- 一个warp是32个线程，是线程调度的基本单位，所以一般情况下每个block的线程数取128，256,512,1024等32的倍数

#### Automatic Scalability
![](image/Pasted%20image%2020250515161011.png)

### 线程同步

**1. Block内的同步**

CUDA 提供原生支持，通过内置函数：
```C++
__syncthreads();
```
来实现阻塞block内所有线程，直到所有线程执行到此处为止

**2. 跨Block的同步**

CUDA 不支持kernel中跨 block 的线程同步，因为每个 block 是独立调度的，它们可能不同时启动、运行在不同 SM 上。

>vLLM中的实现:Global Memory + 原子操作 手动实现 barrier
使用每个block的Thread 0与其他block的线程进行通信，block内部通过`__syncthreads()`实现同步从而完成跨block的同步。
```C++
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg, int rank) {
	uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
	if (threadIdx.x < ngpus) {
		auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
		auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
		// Write the expected counter value to peer and wait for correct value
		// from peer.
		st_flag_volatile(peer_counter_ptr, flag);
		while (ld_flag_volatile(self_counter_ptr) != flag);
	}
	__syncthreads();
	// use one thread to update flag
	if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}
```



---
### CUDA Stream

**1. 并发性**
- 不同的 Stream 可以**并发执行**其队列中的操作。这意味着当一个 Stream 中的某个操作正在 GPU 上执行时，另一个 Stream 中的操作也可以同时进行，从而提高 GPU 的利用率。
- 同一个 Stream 内部的操作是**按添加顺序 (FIFO)** 执行的，一个操作完成后才会开始下一个操作。

**2. 异步性**
- 将操作添加到 Stream 中通常是**异步的**，这意味着 CPU 在将任务添加到 Stream 后会立即返回，而不会等待 GPU 上的操作完成。这使得 CPU 可以继续执行其他任务，例如准备下一个要发送给 GPU 的数据或启动其他 Stream 中的操作。
#### Default Stream
- 当您在没有显式指定 Stream 的情况下执行 CUDA 操作时，CUDA 运行时会使用一个**默认 Stream (Stream 0 或 Null Stream)**。

**使用 Stream 可以**
- **Overlapping：** 通过重叠 CPU 和 GPU 的执行，以及在 GPU 上并发执行不同的任务，可以显著提高程序的整体性能。例如，您可以同时进行数据传输和内核计算。
- **更大的灵活性：** 将多个逻辑独立的任务放入不同stream中并发执行，提高GPU资源利用率

 > DeepSeek的Decoding阶段Stream图，stream 7用于计算， stream 16用于 EP通信
![](image/Pasted%20image%2020250518234424.png)
#### 通信与计算的Overlapping

[内存拷贝与计算Overlapping简单案例](https://github.com/olcf/cuda-training-series/tree/master/exercises/hw7)

将计算任务分为多个chunk，使用多个stream，相邻stream之间使用不同的stream
```C
  for (int i = 0; i < chunks; i++) { 
    cudaMemcpyAsync(cudaMemcpyHostToDevice, streams[i % num_streams]);
    gaussian_pdf<<<((ds / chunks) + 255) / 256, 256, 0, streams[i % num_streams]>>>(d_x + i * (ds / chunks), d_y + i * (ds / chunks), 0.0, 1.0, ds / chunks);
    cudaMemcpyAsync(**, cudaMemcpyDeviceToHost, streams[i % num_streams]);
  }

```


---

### CUDA Graph是什么

CUDA Graph允许将一系列 GPU 操作定义为一个**有向无环图 (DAG)**。这个图可以被捕获并作为一个独立的单元进行多次执行，从而显著减少 CPU 开销并提高性能，尤其是在重复执行相同操作序列的场景下。
CUDA Graph是对于执行在CUDA stream上的异步操作的

CUDA Graph的node可以是任何异步的CUDA操作，包括：
- **CUDA kernel**
- **CPU Function Call**
- **Memcpy/Memset/Memory Alloc/Free**
- **Sub-Graph**
#### 主要优势：

1. **减少 CPU 的 Launch Overhead**：
    - 在传统的 CUDA 编程中，每一个核函数的启动或内存拷贝操作都需要 CPU 发送一个指令到 GPU。当有大量短小的 GPU 操作时，CPU 启动这些操作的开销可能甚至会大于实际kernel执行时间，成为性能瓶颈。
    - CUDA Graph 解决了这个问题。通过将一系列 GPU 操作捕获为一个图，CPU 只需要一次性地提交整个图的执行指令，极大地减少了 CPU 和 GPU 之间的交互次数，从而降低了启动开销。这对于短时、频繁执行的核函数尤其有效。
	>Launch Overhead Reduction
	>![](image/Pasted%20image%2020250520150430.png)

2. **提高 GPU 利用率和吞吐量：**
    - 由于减少了 CPU 的干预，GPU 可以更高效地连续执行图中的操作，减少了 GPU 处于空闲状态的时间，从而提高了 GPU 的利用率和整体吞吐量。
3. **优化执行效率：**
    - 当一个图被捕获后，CUDA 驱动程序可以获得所有操作及其依赖关系的完整描述。这使得驱动程序能够对图的执行进行更深层次的优化，例如调整操作的顺序、合并不必要的同步点等，以实现更优的执行效率。此外，还能进行Graph内的Address Reuse，使用生命周期不重叠的虚拟内存地址。
    > 虚拟地址重用
![](image/Pasted%20image%2020250520001515.png)



#### Pytorch 使用 CUDA Graph
[Accelerating Pytorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
```Python
# capture
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
	static_y_pred = model(static_input)
	static_loss = loss_fn(static_y_pred, static_target)
	static_loss.backward()
	optimizer.step()

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]
for data, target in zip(real_inputs, real_targets):
	# Fills the graph's input memory with new data to compute on
	static_input.copy_(data)
	static_target.copy_(target)

	g.replay()
```

#### CUDA Graph的局限性

- **图的结构相对静态:** 一旦图被捕获或构建完成，其基本结构（节点和依赖关系）在后续的执行中通常是固定的。如果操作序列或依赖关系发生显著变化，可能需要重新创建图。
- **首次执行的开销:** 创建和实例化 CUDA Graph 需要一定的开销。只有当图被多次重复执行时，才能摊销这个初始开销，并体现出性能优势。

例如在LLM inference中，以vLLM为例，只在Decoding阶段使用了CUDA Graph，因为prefill阶段的输入长度变化幅度很大，而Decoding阶段可以为常用的batch预构建CUDA Graph以加快推理速度

---
## Triton

### GPU编程的挑战

在优化 CUDA 代码时，需要重点关注现代 GPU 的三大核心组件及其带来的挑战：

- **DRAM：**
    - 内存传输速度。必须将数据访问合并成大型事务，以充分利用DRAM的大总线宽度，提高数据吞吐量。
- **SRAM ：**
    -  数据复用与访问冲突。需要**手动管理**数据在SRAM中的存储，以供重复使用；同时要小心避免**共享内存 bank 冲突**，确保高效检索。
- **ALUs：**
    - 计算分配与并行效率。必须**仔细规划**计算任务在不同 SM之间及内部的划分和调度，以最大化指令级/线程级并行性，并充分利用**专用ALU**的性能。

### Triton的优势

1. 自动调优
在第一次运行时Triton会在多个给定的参数下运行，确定最佳的`BLOCK_SIZE`
2. Automatically schedule
自动进行SMs内的调度，自动管理`Shared Memory`
无需进行SM内的调度，但是
>Triton makes it possible to reach peak hardware performance with relatively little effort; for example, it can be used to write FP16 matrix multiplication kernels that match the performance of cuBLAS—something that many GPU programmers can’t do—in under 25 lines of code.
### CUDA与Triton的对比

在 **CUDA 编程中，HBM（全局内存）中的数据访问是隐式的**，你只需要使用指针或数组访问即可，不需要显式调用像 `tl.load()` 这样的函数。但在 **Triton 中，`tl.load()` 是必须显式调用的**，因为 Triton 是基于 MLIR 的中间表示系统，需要你显式描述内存访问。
![](image/gpu-architecture.svg)

| 特性        | **Triton**                                                    | **CUDA**                  |
| --------- | ------------------------------------------------------------- | ------------------------- |
| **开发者体验** | 高层、类 Python，学习曲线较低                                            | 类 C++，复杂度更高               |
| **抽象级别**  | 高：屏蔽线程块、warp、shared memory 等底层细节                              | 低：需要手动管理线程、内存等            |
| **灵活性**   | 针对深度学习场景做了优化，适用于大多数算子优化                                       | 灵活全面，可开发任意类型的 GPU kernel  |
| **性能**    | 在某些特定深度学习算子上性能接近甚至超过 深度调优的CUDA  kernel 性能                     | 最强性能潜力，但需要大量调优            |
| **支持平台**  | NVIDIA GPUs (Compute Capability 8.0+)  & AMD GPUs (ROCm 6.2+) | 主要支持 NVIDIA GPU           |
| **生态**    | 新且小众，生态仍在发展                                                   | 成熟、广泛使用，文档丰富              |

### 工作原理


- 当用 Triton 编写的内核首次运行时，Triton 的自动调优器会尝试不同的组合，运行少量迭代来测量它们的实际执行时间。
- 调优过程结束后，Triton 会将最佳参数组合缓存起来。后续再次运行相同的内核时，它会直接使用缓存中的最佳参数，无需重新调优。

### Autotune

```Python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config(kwargs={'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
        # ... 更多配置
    ],
    key=['input_size'] # 当 input_size 变化时重新调优
)
@triton.jit
def my_kernel(input_size, ...):
    # Kernel code
    pass
```
使用`@triton.autotune`预定义一系列Config，triton会多次运行并测量其在目标硬件上的执行时间，选择执行时间最短的配置作为当前`key`的最佳配置并缓存。
当输入参数`key`改变时，Triton会认为当前的工作负载可能已经不同，需要重新运行调优过程以找到新的最佳配置。

>vLLM中的Triton Config，预定义了一系列在特定硬件一系列batch size下的最优参数
>![](image/Pasted%20image%2020250520202736.png)



### Triton Kernel

编写Triton kernel的步骤和CUDA类似，但是省去了处理Thread的步骤，只需处理一个Block内的计算逻辑。一般可以分为3个步骤
1. 使用`tl.program_id(axis=0)`获取pid并根据pid确定该Block对应处理的位置，并计算出指针偏移量
2. 使用`tl.load()`将数据逐chunk加载进SRAM，随后进行计算
3. 使用`tl.store()`将计算结果保存到对应地址


这里我使用了一段profile代码来分析Triton的执行流程

Triton调用代码片段：
```Python
a = torch.randn((2048 * 6, 10240), device=DEVICE, dtype=torch.float16)
b = torch.randn((10240, 2048), device=DEVICE, dtype=torch.float16)
profile_dir = "/home/xxx/Desktop/profile"

num_iteration = 2

with torch.profiler.profile(
	activities=[
		torch.profiler.ProfilerActivity.CPU,
		torch.profiler.ProfilerActivity.CUDA
	],
	on_trace_ready=torch.profiler.tensorboard_trace_handler(
						str(profile_dir)),
) as p:

for i in range(num_iteration):
	with torch.profiler.record_function(f"i_{i}"):
		triton_output = matmul(a, b)
```

这里将num_iterations设置为2，`matmul`只会执行两次，使用`torch.profiler.record_function`标记这两次调用，得到结果如下：
> i_0 和 i_1 整体
![](image/Pasted%20image%2020250521001122.png)

左边蓝色部分的是`i_0`即第一次调用，右边绿色部分是第二次调用，可见Triton的autotune在初次调用会多次尝试不同的`block_size`，选取最佳的启动参数。

> i_1 profile细节
![](image/Pasted%20image%2020250521002004.png)

观察`i_1`下面的曲线发现，即便是之调用了一个Triton Kernel，也会启动多个CUDA Kernel



### L2 Cache Optimization案例

Triton官方的Tutorials中有关于L2 Cache Optimization的内容，vLLM的FusedMoE的Triton实现直接照搬了其中的前几行代码
>经典行优先和Grouped Optimization的pid到输出矩阵C Block的映射，这里的Group size为2
![](image/Pasted%20image%2020250521004904.png)

这样做的好处：使GPU一次能加载到SRAM中执行的块尽可能地呈一个方形，减少同一时间需要加载进SRAM的不同的行或列数。
![](image/Pasted%20image%2020250521004350.png)

### Triton的局限性

