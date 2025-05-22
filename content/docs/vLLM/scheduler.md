---
title: scheduler
---


优先级顺序：
1. swaped
2. prefill
3. runninng (Decoding and Chunked Prefill)

调度时，如果有被换出的任务（swaped），则优先将swaped考虑，
如果没有未完成的prefill任务，优先将prefill任务加入
最后考虑running的任务

代码如下：
```bash
vllm/core/scheduler.py

class Scheduler:
	def _schedule_defult(self) -> ScheduleOutputs
		prefills = SchedulerPrefillOutputs.create_empty()
		running_scheduled = SchedulerRunningOutputs.create_empty()
		swapped_in = SchedulerSwappedInOutputs.create_empty()
		# 如果没有swaped才考虑prefill
		if not self.swapped:
			prefills = self._schedule_prefills(budget,
			curr_loras,
			enable_chunking=False)
		# 如果没有prefill才考虑running
		if len(prefills.seq_groups) == 0:
			running_scheduled = self._schedule_running(budget,
			curr_loras,enable_chunking=False)
			if (len(running_scheduled.preempted) +
				len(running_scheduled.swapped_out) == 0):
				swapped_in = \
				self._schedule_swapped(budget, curr_loras)

		# 将被抢占的任务加到waiting队列
		self.waiting.extendleft(running_scheduled.preempted)
		# 添加新的prefill请求
		if len(prefills.seq_groups) > 0:
			self.running.extend([s.seq_group for s in prefills.seq_groups])
		
		self.running.extend(running_scheduled.decode_seq_groups_list)
		if len(swapped_in.decode_seq_groups) > 0:
			self.running.extend(
				[s.seq_group for s in swapped_in.decode_seq_groups])

		# Update swapped requests.
		self.swapped.extend(running_scheduled.swapped_out)
		preempted = len(running_scheduled.preempted) + len(
		running_scheduled.swapped_out)
```

上述代码中的关键点：*budget* 
budget可以理解为一个token预算，即系统能处理的token量，添加任务会减少budget，任务完成之后会归还budget，budget是一个动态的表示设备容量的*桶* ，

_schedule_default函数还会调用下面三个函数
- _schedule_prefills
- _schedule_running
- _schedule_swaped
每次调用时会消耗掉一定的budget



-------

scheduler拥有正在running的任务的引用即self.running
在调度正在运行的任务过程中，会running遍历running队列中的元素并返回，


