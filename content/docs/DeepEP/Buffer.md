---
title: Buffer
---

DeepEP中的Buffer是其操作的核心，需要在初始化时被创建一次，而不是在每次调用dispatch和combine时创建

## Device的Buffer
定义于文件`buffer.cuh`中
当在Device中调用时，不是cudaMemalloc，只是简单的改变指针偏移量
比如这里的：
```Cpp
auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
auto channel_end_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
```
都是在kernel中调用的，不会出现实际的内存分配，调用的是下面的逻辑：
```Cpp
__device__ __forceinline__ Buffer(void* &gbl_ptr, int num_elems, int offset = 0) {
total_bytes = num_elems * sizeof(dtype_t);
ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + offset * sizeof(dtype_t);
gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
}
```
这里使用了指针引用，改变了局部变量指针的指向，并将指针保存到成员变量`ptr`中


## Host的Buffer
定义于`deep_ep.hpp`中，是面向调用者的Buffer类
初始化逻辑定义在文件`deep_ep.cpp`的第一个函数，包含cudaMalloc，是对整个buffer的分配

---
## Buffer 布局
![](image/Pasted%20image%2020251014224803.png)

