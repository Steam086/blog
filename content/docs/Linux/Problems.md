---
title: 已知的问题
---

## wayland下连接显示器无法使用高刷（100Hz）
2025/4/22
将刷新率调整到100Hz之后外接的 显示器会黑屏
NVIDIA显卡


## wayland下使用微信Linux版(QT)缩放异常
2025/5/22
窗口很小
>**已经解决**，启动时加上参数：env QT_AUTO_SCREEN_SCALE_FACTOR=1 QT_QPA_PLATFORM=xcb QT_IM_MODULE=fcitx 
>可以直接修改wechat.desktop
## Jetbrains的IDE无法使用输入法
2025/4/22
软件均更新到最新版本

