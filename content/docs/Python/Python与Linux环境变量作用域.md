
## Case 1-使用export
假设有一个shell脚本如下：
```
export A=1
export B=2
python env_test.py
```

Python脚本中的内容如下
```Python {filename=env_test.py}
import os

a = os.environ.get("a")
b = os.environ.get("b")
print(a)
print(b)
```
输出为：
```
1
2
```

### Case 2-使用Python多进程

shell脚本同case1，Python脚本如下
```Python
import os
from multiprocessing import Process

def foo():
	a = os.environ.get("A")
	b = os.environ.get("B")
	print("new proc"+a)
	print("new proc"+b)
  

p = Process(target=foo)
p.start()
p.join()
```
脚本输出是：
```
new proc1
new proc2
```
所以使用Python的多进程，新创建的进程会继承原进程的环境变量

## Case 3-在ray集群中

ray集群中启动的worker只能读取到ray start之前的所有环境变量

例如，有以下脚本：
```shell
ray start --address="{ip}:{port}"
export VAR=1
```
此时ray集群中运行的进程无法获取到环境变量VAR

比如在vllm中，如果想使用VLLM的Pytorch profiler功能，需要设置环境变量`VLLM_TORCH_PROFILER_DIR`这个环境变量，但是如果使用ray集群进行多node部署，必须在每个节点的ray脚本之前设置这个环境变量。