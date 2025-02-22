使用vlllm的命令行部署的全流程：

`vllm serve <model_tag> [options]`
执行上述命令时，入口是vllm/scripts.py
- 进入scripts.py 中的main函数
- 将所有的命令行参数通过`argparser`输入到`args`中
- 进入serve函数 `uvloop.run(run_server(args))`
- 在`run_server`函数中经过一系列的条件判断，执行`async with build_async_engine_client(args) as engine_client:`，根据命令行参数构建`engine_client`
- `async def init_app_state`函数根据config初始化