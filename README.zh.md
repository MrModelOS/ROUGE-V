<h1 align="center">ROUGE-V</h1>

<p align="center">一个将 CUDA PTX 翻译为 RISC-V 原生代码的编译器。</p>

<p align="center">
<a href="https://github.com/MrModelOS/ROUGE-V/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/MrModelOS/ROUGE-V/ci.yml?style=flat-square&branch=main"></a>
<img alt="测试" src="https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen?style=flat-square">
<img alt="语言" src="https://img.shields.io/badge/language-C%2B%2B-blue?style=flat-square">
<img alt="许可证" src="https://img.shields.io/badge/license-Apache--2.0%20WITH%20LLVM--exception-blue?style=flat-square">
</p>

<p align="center">
<a href="README.md">English</a> ·
<a href="README.ru.md">Русский</a> ·
<a href="README.zh.md">中文</a>
</p>

---

`nvcc` 并不直接为 GPU 生成机器码。它生成的是 **PTX** —— 一套有公开规范的文本汇编。
任何能读懂 PTX 的工具，都可以把它编译到别的目标平台上。

ROUGE-V 做的正是这件事：PTX → LLVM IR → 原生代码，面向 RISC-V 及其向量扩展
`rv64gcv`。这是编译器，不是模拟器：翻译后的内核是被编译出来的，而不是逐条指令
解释执行的。

### 安装

```sh
git clone https://github.com/MrModelOS/ROUGE-V.git
cd ROUGE-V
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

需要 CMake ≥ 3.20、支持 C++17 的编译器和 `clang`。`nvcc` 是可选的 —— 装上它
才会启用厂商 PTX 测试。

### 使用

```sh
./build/rouge-compiler/ptx2ir \
    software/rouge-compiler/tests/kernels/vadd.ptx /tmp/vadd.ll

clang -O2 -c /tmp/vadd.ll -o /tmp/vadd.o                  # 主机
clang --target=riscv64-unknown-elf -march=rv64gcv \
      -c /tmp/vadd.ll -o /tmp/vadd-rv64.o                 # RISC-V + Vector
```

目标三元组由翻译器一侧的 `--target` 决定 —— 它只是 `ptx2ir` 参数的前缀，
其余命令不变。同一个 PTX 可以为四个平台构建：

```sh
./build/rouge-compiler/ptx2ir --target amdgcn-amd-amdhsa kernel.ptx kernel-amd.ll
clang --target=amdgcn-amd-amdhsa -mcpu=gfx1100 -c kernel-amd.ll -o kernel-amd.o
```

| `--target` | 平台 | 后端参数 | 实际验证到的内容 |
|---|---|---|---|
| *(默认)* `x86_64-pc-linux-gnu` | x86-64 主机 | —— | **构建并执行** —— `aot_native_*` ctest 套件 |
| `riscv64-unknown-elf` | RISC-V + Vector | `-march=rv64gcv` | 构建为原生目标文件 —— `ctest compiler_rvv_backend_*` |
| `amdgcn-amd-amdhsa` | AMD RDNA 3 | `-mcpu=gfx1100` | 构建为原生目标文件 —— 从未执行 |
| `nvptx64-nvidia-cuda` | NVIDIA | `-march=sm_75` | 构建为原生目标文件 —— 从未执行 |

`software/rouge-compiler/tests/kernels/` 中全部五个标准内核 —— `vadd`、
`block_reduce`、`atomic_reduce`、`fp16_reduce`、`gemm_tile` —— 都能在四种
三元组下构建。

两行 GPU 是**构建层面**的结论，仅此而已。项目中没有 AMD 或 NVIDIA 设备，
因此这两行没有任何内容被实际执行，也无法由此得出任何性能数字。真正的执行
只在 x86-64 上验证过；RISC-V、AMD 和 NVIDIA 验证的是真实 LLVM 后端能够接受
所生成的目标文件。目标选择会改变所输出 IR 中的地址空间，这正是这些后端能够
生成 `global_load`/`global_store` 而非标量访存的原因 —— 详见
[docs/06-compiler-architecture.md](docs/06-compiler-architecture.md)。

```
$ ctest --test-dir build
...
 5/16 aot_native_block_reduce ..............   Passed   smem 1024B 16x256
 6/16 aot_native_atomic_reduce .............   Passed   sum=4608 count=4096
 7/16 aot_native_fp16_reduce ...............   Passed   32x32 f16/bf16
 8/16 aot_native_gemm_tile .................   Passed   2x2 blocks 16x16
13/16 compiler_nvcc_ptx ....................   Passed   nvcc -> ptx2ir -> clang
100% tests passed out of 16
```

> [!NOTE]
> `compiler_nvcc_ptx` 用厂商自带的 `nvcc` 编译真实的 CUDA C，并把生成的 PTX
> 推过整条流水线。未安装 `nvcc` 时该测试会被跳过。

> [!IMPORTANT]
> 不在支持子集内的指令会直接报错，并指出是哪一条指令。不存在静默的错误编译
> —— 给出错误答案比拒绝执行更糟。

### 组件

| 组件 | 说明 |
|---|---|
| [`rouge-ptx`](software/rouge-ptx/) | 共享 PTX 前端：解析器与参考解释器 |
| [`rouge-compiler`](software/rouge-compiler/) | `ptx2ir`，AOT 翻译器；宿主运行时；MLIR 轮廓 |
| [`rouge-cuda`](software/rouge-cuda/) | CUDA Driver/Runtime API 拦截与备用执行路径 |

两条执行路径建立在同一个解析器之上。这是有意为之：同一套语法，同一套关于寄存器、
参数、`.shared` 布局和标号的规则，从而使两条路径不会产生分歧。测试套件正是围绕
这一要求构建的 —— 在每个内核上，两条路径必须给出**逐位相同**的结果。

### 支持的 PTX 子集

| 分组 | 指令 |
|---|---|
| 参数 | `ld.param.{u32,u64}` 及有符号/无符号位变体 |
| 传送 | `mov.*`、`cvta.to.global.*`、`cvta.to.shared.*` |
| 算术 | `add` `sub` `mul` `div` `rem`、`mad.lo` `mad.hi`、`mul.wide.*` |
| 位运算 / 移位 | `and` `or` `xor` `not`、`shl`、`shr.{u,s}`、`bfi` `bfe` |
| 比较 | `setp.{ge,gt,lt,le,eq,ne}.{u32,s32,f32}` |
| 选择 | `selp.*`、`slct.*`、谓词 `and` `or` `not` |
| 最值 / 一元运算 | `min` `max.*`、`neg`、`abs`、`popc`、`clz`、`bfind.shiftamt` |
| 浮点 | `f16` / `bf16` 运算、`fma.rn`、`sqrt`、`rcp`、`rsqrt`、`fneg` |
| 访存 | global 与 shared 的 `ld`/`st`：`f32` `u32` `u64` `u8`/`s8` `u16`/`s16` `f16` `bf16` `v2.f32` |
| 共享内存 | `.shared` 声明、`cvta.to.shared`、`[reg+offset]` 寻址 |
| 同步 | `bar.sync` → 运行时屏障 |
| Warp | `shfl.{idx,bfly,up,down}`（含带谓词的目的寄存器）、`vote.*`、`activemask` |
| 原子操作 | `atom.add` `cas` `exch` `min` `max`、`red.add.*` → `atomicrmw` / `cmpxchg` |

张量指令（`mma`）、`f16x2` 向量形式和 `cvt.sat` 舍入模式尚未实现，会被报告为错误。

### 文档

[docs/06-compiler-architecture.md](docs/06-compiler-architecture.md) —— 启动描述符、
共享内存布局、`bar.sync` 模型、原子操作、FP16/BF16 的降低，以及规划中的
SIMT → RVV 路径。

### 基于 ROUGE-V 的项目

如果你维护着使用 ROUGE-V 的项目 —— fork、绑定、后端等 —— 请在你的 README 中
注明：该项目并非由 ROUGE-V 团队开发，与其无隶属关系。

### 法律声明

ROUGE-V 不包含任何 NVIDIA 的源代码、头文件或二进制文件，也不对其进行逆向工程。
它所翻译的 PTX 由用户自己的 `nvcc` 生成，依据用户持有的 NVIDIA 许可证，并遵循
已公开的规范。`software/rouge-cuda/include/` 下的 CUDA API 头文件是根据公开的
API 文档从零编写的。clean-room 规则记录在 [CONTRIBUTING.md](CONTRIBUTING.md) 中。

ROUGE-V 与 NVIDIA 无隶属关系，未获其认可或支持。CUDA 及其他商标归各自所有者
所有，此处仅用于描述兼容性。

### 参与贡献

请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md) —— 其中说明了构建方式、测试流程，
以及适用于所有贡献的 clean-room 要求。Issue 与 pull request 均开放。

### 致谢

- **LLVM / MLIR** —— 本项目所依托的 IR、后端与 pass 基础设施
- **RISC-V International** —— `rv64gcv` ISA 与 Vector Extension 规范
- **NVIDIA** —— 界定了输入格式的公开 PTX ISA 规范与 CUDA API 文档

### 许可证

Apache 2.0 WITH LLVM-exception —— 与 LLVM 相同的许可证，使本项目代码可以
无障碍地集成进 LLVM 与 MLIR。
