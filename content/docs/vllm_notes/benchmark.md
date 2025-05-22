---
type: docs
draft: true
title: Benchmark
---

### 1. 在统计kernel执行时间时，不要使用time.time()来计算

time.time()得到的只是大概的CPU时间，不能真实反映kernel的执行时间

注意事项
- 要有warm up
- 必须用 `torch.cuda.synchronize()`