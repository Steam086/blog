

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy
## 2.3. Host Compiler Support Policy[](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy "Permalink to this headline")

In order to compile the CPU “Host” code in the CUDA source, the CUDA compiler NVCC requires a compatible host compiler to be installed on the system. The version of the host compiler supported on Linux platforms is tabulated as below. NVCC performs a version check on the host compiler’s major version and so newer minor versions of the compilers listed below will be supported, but major versions falling outside the range will not be supported.

Table 2 Supported Compilers[](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#id61 "Permalink to this table")

|Distribution|GCC|Clang|NVHPC|XLC|ArmC/C++|ICC|
|---|---|---|---|---|---|---|
|x86_64|6.x - 14.x|7.x - 19.x|24.9|No|No|2021.7|
|Arm64 sbsa|6.x - 14.x|7.x - 19.x|24.9|No|24.04|No|

For GCC and Clang, the preceding table indicates the minimum version and the latest version supported. If you are on a Linux distribution that may use an older version of GCC toolchain as default than what is listed above, it is recommended to upgrade to a newer toolchain CUDA 11.0 or later toolkit. Newer GCC toolchains are available with the Red Hat Developer Toolset for example. For platforms that ship a compiler version older than GCC 6 by default, linking to static or dynamic libraries that are shipped with the CUDA Toolkit is not supported. We only support libstdc++ (GCC’s implementation) for all the supported host compilers for the platforms listed above.