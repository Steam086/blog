---
title: Fedora与Ubuntu
date: 2025-07-04T16:36:38+08:00
---
对于一个简单的CUDA Hello World程序的编译，使用Fedora和Ubuntu的区别：


## Fedora

## Step 1:

Download Nvidia Driver via dnf:
```bash
sudo dnf update -y
sudo dnf install akmod-nvidia
```

## Step 2:

Download Nvidia CUDA 
https://developer.nvidia.com/cuda-toolkit
Select version of CUDA and download
Install CUDA
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run
```

## Step 3:

CUDA only support gcc version lower than 14.x, so you should download the correct version of gcc and glibc.

download gcc: https://gcc.gnu.org/releases.html
download glibc: 


