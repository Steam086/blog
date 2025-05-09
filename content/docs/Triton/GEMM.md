GEMM公式：
$$C = \alpha A \cdot B +\beta C$$
简化处理这里仅讨论
$$C =  A \cdot B $$

函数参数说明：
- a_ptr: 矩阵A的地址
- b_ptr: 矩阵B的地址
- c_ptr: 结果矩阵C的地址
- BLOCK_SIZE_M, BLOCK_SIZE_N: 将矩阵乘法分块，一次要处理的块大小
>
>调用kernel时传入的`grid`是将任务分配的方式，可以是一维或者多维的，这里只传入了一维的
>kernel函数中使用`tl.program_id` 来获取pid，以便程序知道自己要处理哪一块区域

### 1. 映射pid到它要计算的矩阵 C 的block

与传统的行优先算法不同，Grouped算法将A的行分为多个组，这里展示了Group_size = 2的情况，pid与其要计算的块对应方式如下：
![[Pasted image 20250423204527.png]]
映射完成后，根据pid对应的块算出`a_ptrs`和`b_ptrs`偏移量。

> Note
>行优先和带L2 Cache优化的pid映射区别如下：
>![[Pasted image 20250424101740.png]]
>即将多个行合并为一个group，在group内进行列优先，使得在组内的相邻块之间使用相同的B block。
>这样做的好处是可以减少需要加载到L2 Cache的block，同样是并行计算4个block，左边的方法需要将$A[0,:] ,B[:,0:3]$加载进缓存（共5个block），但是右边只需要加载$A[0:1,:],B[:, 0:1]$加载4个block。


### 2. 根据K维度的BLOCK_SIZE_K获取a_ptrs和b_ptrs

`a_ptrs`和`b_ptrs`的shape分别为$(BLOCK\_SIZE\_M, BLOCK\_SIZE\_K)$和$(BLOCK\_SIZE\_K, BLOCK\_SIZE\_N)$ ，指向循环中第一个要加载的block中的所有元素

在计算`a_ptr`过程中，用到了一个broadcast的小trick， 将两个向量维度扩张之后相加，生成一个矩阵：
```Python
offs_k = tl.arange(0, B_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
```
`offs_k`扩张后的shape是 `(1, B_K)`
`offs_am`扩张后的维度是`(B_M, 1)`
两者相加之后的维度是`(B_M, B_K)`
### 3. for循环启动
- 将a_ptrs和b_ptrs中所有元素加载到**SRAM**
- 计算tl.dot(a,b)，并将结果存入acc
a和b沿K dim方向前进，在越界前循环结束
```C++
a_ptrs += BLOCK_SIZE_K * stride_ak
b_ptrs += BLOCK_SIZE_K * stride_bk
```
### 4. 数据类型转换并将c写回HBM对应地址
- 计算要写入的C的地址
- `tl.store`将acc写入对应的块