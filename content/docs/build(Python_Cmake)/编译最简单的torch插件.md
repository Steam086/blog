---
title: 简单的torch operator编译
date: 2025-06-27T16:36:38+08:00
---
本样例参考了vLLM源码仓和Pytorch官方文档

## Python调用代码

```Python
import torch

torch.ops.load_library("./_C_TORCH.so")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
result = torch.ops._C.mymuladd(a, b)
print(result) # Expected output: tensor([ 4., 10., 18.])
```

## C++ 代码

简单起见，这里只用到了CPU的实现，正常定义函数，然后使用TORCH_LIBRARY定义，注意这里micro的第一个参数是namespace,即在torch中调用必须使用这里传入的字符串。
必须像这样使用
`torch.ops._C.函数名`

```Cpp
#include <torch/extension.h>

torch::Tensor mymuladd(torch::Tensor a, torch::Tensor b) {
	TORCH_CHECK(a.device().is_cpu(), "Tensor a must be on CPU");
	TORCH_CHECK(b.device().is_cpu(), "Tensor b must be on CPU");
	TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same shape");
	return a * b ;
}

TORCH_LIBRARY(_C, m) {
	m.def("mymuladd(Tensor a, Tensor b) -> Tensor", &mymuladd);
}
```


## CMakeList.txt代码

这里最重要的是要让cmake找到正确的Python路径和Torch的路径
有
list(APPEND CMAKE_PREFIX_PATH "/home/jjr/pyenvs/vllm-py312/.venv/bin/")
list(APPEND CMAKE_PREFIX_PATH "/home/jjr/pyenvs/vllm-py312/.venv/lib/python3.12/site-packages/torch/share/cmake")
可以将Python和torch引入cmake中，后面可以使用`Python_add_library`取代add_library

```cmake
cmake_minimum_required(VERSION 3.10)
project(vllm_extensions LANGUAGES CXX)
list(APPEND CMAKE_PREFIX_PATH "/home/jjr/pyenvs/vllm-py312/.venv/bin/")
list(APPEND CMAKE_PREFIX_PATH "/home/jjr/pyenvs/vllm-py312/.venv/lib/python3.12/site-packages/torch/share/cmake")
set(CXX_STANDARD 11)
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
find_package(Torch REQUIRED)
set(VLLM_EXT_SRC "csrc/torch_bindings.cpp")
Python_add_library(_C_TORCH MODULE "${VLLM_EXT_SRC}")
target_include_directories(_C_TORCH PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(_C_TORCH PRIVATE torch)
set_target_properties(_C_TORCH PROPERTIES CXX_VISIBILITY_PRESET hidden)
```
### configure传入的参数
![](image/Pasted%20image%2020250627160245.png)

