## weight_loader

weight由model自行实现
- `weight_loader`：本质是一个将已经处在内存中（可能是一个`Generator[]`）的模型参数加载到对应model类的`self.param`中。
例如，张量并行中，每个GPU只享用一部分模型参数，只要将需要的部分模型参数加载到参数列表中即可。
- 每个需要加载模型参数的nn.Module都 要么**自定义weight_loader**，要么使用继承来的weight_loader，确保模型参数能够按照正确的方式加载到GPU中。
比如，要想实现对已经下载的模型权重的量化，可以自定义weight_loader

## model_loader

由文件格式决定
负责将不同文件格式的model从硬盘加载到内存中，若模型不存在，则从Hugging Face或ModelScope下载


## Question

1. pipeline parallelism是如何实现的？？？（vllm中只有部分模型实现了流水线并行）
2. kv_tarns是怎么进行的
3. 多线程是从调用的哪一步开始的

