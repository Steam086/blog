---
title: Memory Fence
---

## Acquire Release

在生产者消费者模型中，生产者向buffer中写入数据并更新tail，消费者轮询tail之后读取数据，要用到如下代码：

生产者：
```cpp
write(buffer)
write(tail) //表示已经写入完成
```

消费者：
```cpp
while(tail == cached_tail);
read(buffer)
```

此时由于CPU的乱序执行，导致可能对tail的写入先于对buffer的写入，这在多线程协作时是不被允许的，所以要在`write(buffer)`与`write(tail)`之间加上内存屏障。
确保write(tail)之前的任何操作都不能越过tail，这里需要加上**load store**和**store store**屏障，
```cpp
write(buffer)
# load store 两个屏障确保不会越界
# store store 
write(tail) //表示已经写入完成
```

对于消费者，要确保不能在读取tail之前读取buffer，需要设置内存屏障
load load 屏障和 load store屏障
```cpp
while(tail == cached_tail);
# load load 
# load store
read(buffer)
```
确保在读取到tail之前不能进行任何store和load的操作

这两种内存屏障通常成对出现，被称为acquire和release语义
现在通常在读写这种标记变量时使用acquire和release语义

生产者使用release语义写入，消费者通过acquire语义读取标记，用来保证并发的确定性

生产者：
```cpp
write(buffer)
release_store(tail)
```
消费者：
```cpp
while(acquire_load(tail) == cached_tail);
read(buffer)
```

 >DeepEP中广泛使用这种方式来控制缓冲区的读写