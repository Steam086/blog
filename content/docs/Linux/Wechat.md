---
title: Wechat缩放和输入法
---
## 微信Linux版是QT开发的

在我的电脑上正常下载微信的rpm包并安装之后，微信的缩放比例会很奇怪（可能是wayland的原因）而且不能使用输入法，可以使用设置环境变量的方式启动微信。

在启动脚本或者.desktop文件中加入：

```shell
QT_AUTO_SCREEN_SCALE_FACTOR=1
QT_QPA_PLATFORM=xcb
QT_IM_MODULE=fcitx
```
启动自动缩放并将输入法正确启动。
>[!important]
>注意不要在.bashrc中添加，要在微信的启动脚本中添加
