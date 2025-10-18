---
title: intranode
---
## Q:dispatch阶段修改的send_head是什么

```Cpp
if (send_lane_id == 0)
send_head[token_idx * kNumRanks + send_warp_id] = is_token_in_rank[token_idx * kNumRanks + send_warp_id] ? cached_channel_tail_idx : -1;
```

send_head只存在于intra_node的收发中，在dispatch阶段中被修改，在combine阶段中被使用
目的可能是优化收发速度，提前指出