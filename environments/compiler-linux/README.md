# 本机R3私有编译工具链

R3首次正常图执行在正式请求前因缺少C编译器失败。为保留该基线，没有回退eager，也没有更改系统包或Python依赖锁；从Ubuntu官方apt元数据下载固定27包，逐个核验SHA256后仅用dpkg-deb解包到项目私有sysroot。

本机登记根为$HOME/.local/share/llm-serving-lab/compiler-gcc14-r3-20261103。包总73,695,152字节；GCC14.3.0，复用已装系统binutils2.46。C/C++共享库编译、链接和ctypes加载已通过，dpkg包清单前后相同。包版本和摘要见package-manifest.json，构建步骤见bootstrap.py。

工具链只供当前Ubuntu26.04/x86_64本地环境。bootstrap使用本机已核apt包版本，不代表任意发行版可直接安装；换机器须重新核对发行版、所有权和兼容性。脚本不执行系统包安装或维护脚本。首次链接发现leaf包不携带基础系统的usrmerge布局，已仅在私有sysroot添加相对lib/lib64/bin/sbin链接；原失败和初版脚本保留在R3工具链工件中。

CC/CXX是单个可执行wrapper，传入私有sysroot和编译器搜索路径；LD_LIBRARY_PATH只作用于编译子进程，生成模块使用明确RPATH。worker的Python环境不被全局改写。启动器记录工具链receipt、manifest/wrapper摘要及私有编译缓存路径。

工具链、源deb及缓存作为后续本项目图执行依赖保留。Python锁、模型参数和旧R1/R2结果不改。A轮启动失败不计为测量结果；B使用新运行身份、相同负载/SLO重新执行。工具链安装不是性能优化成果。
