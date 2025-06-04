---
title: broadcast
---


### 案例引入

torch.argsort()


```Python
import torch

  
x = torch.randn((5, ))
y = torch.arange(0, 10) % 5
x = x[y]
print(x[0:5])
print(x[5:10])
```
输出为
```
tensor([-1.6093, -0.4396, -2.3721,  1.7828, -0.9525])
tensor([-1.6093, -0.4396, -2.3721,  1.7828, -0.9525])
```