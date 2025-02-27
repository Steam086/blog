## Distributed Executor
vllm提供多种后端实现
1. ray-based
2. Python multiprocessing-based
.MultiprocessingDistributedExecutor 好像是只能组织tp
## engine负责组织流水线并行

Detect GTK_IM_MODULE and QT_IM_MODULE being set and Wayland Input method frontend is working. It is recommended to unset GTK_IM_MODULE and QT_IM_MODULE and use Wayland input method frontend instead. For more details see https://fcitx-im.org/wiki/Using_Fcitx_5_on_Wayland#KDE_Plasma


vllm暂时对多节点并行的支持是通过ray实现的
## Running vLLM on a single node
[#](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#running-vllm-on-a-single-node "Permalink to this heading")

vLLM supports distributed tensor-parallel and pipeline-parallel inference and serving. Currently, we support [Megatron-LM’s tensor parallel algorithm](https://arxiv.org/pdf/1909.08053.pdf). We manage the distributed runtime with either [Ray](https://github.com/ray-project/ray) or python native multiprocessing. Multiprocessing can be used when deploying on a single node, multi-node inferencing currently requires Ray.