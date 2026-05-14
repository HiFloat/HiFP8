# HiFP8 — HiFloat8 量化框架

基于 [HiFloat8 (arxiv 2409.16626)](https://arxiv.org/abs/2409.16626) 的量化实现，支持 **BF16 伪量化** 和 **uint8 真量化** 双模式，无需修改 torchao 源码，可通过 vLLM 部署推理。

> **术语说明**：**HiFloat8** 是论文定义的 8-bit 自适应浮点格式；**HiFP8** 是本项目名称；**HiF8** 是 vLLM fork 中的 `quant_method` 标识。

## 特性

- **双模式量化**：BF16 伪量化（训练/校准） + uint8 真量化（部署/压缩）
- **HiF8 导出**：为 [vLLM-HiF8 fork](https://github.com/XiangWanggithub/vllm.git) 导出预量化权重，支持 torch.compile 加速推理
- **非侵入式设计**：所有代码位于 `./ao/` 外部，torchao 源码只读
- **vLLM 原生集成**：通过 v4 server 自动检测量化格式，零配置部署
- **HiFloat8 CUDA 内核**：自定义 8-bit 自适应精度编码/解码，支持 float32/float64/bfloat16 + CPU fallback
- **Evalscope 评测**：ARC / MMLU / CEval 等标准 benchmark 一键评估
- **SmoothQuant 集成**：支持 smooth_scale 导出与 vLLM-HiF8 fork 运行时动态应用
- **MoE 支持**：验证支持 Qwen3-30B-A3B、GPT-OSS-20B 等 MoE 架构

## 精度评测 (ARC Benchmark)

| 模型 | ARC-Easy | ARC-Challenge | Mean | 备注 |
|------|----------|---------------|------|------|
| Qwen3-0.6B (原始) | 0.74 | 0.60 | **0.67** | 100 samples |
| + HiFP8 BF16 伪量化 | 0.74 | 0.60 | **0.67** | 100 samples |
| + HiFP8 uint8 真量化 | 0.73 | 0.66 | **0.695** | 100 samples, 2x 压缩 |
| GPT-OSS-20B HiF8 | 0.9428 | 0.9317 | **0.9391** | 全量评测, torch.compile |

> Qwen3: 100 samples/subset, GPT-OSS: 全量 (ARC-Easy 2376, ARC-Challenge 1172)
> 全部使用 evalscope 评测, vLLM 推理



## 目录结构

```
hifp8/
├── custom_ops/                    # Layer 1: 核心算子
│   ├── hifp8_ops.py               #   伪量化 kernel（NPU 适配替换点）
│   ├── hifp8_uint8_ops.py         #   uint8 编码/解码 + direct fake_quant
│   ├── hifp8_uint8_layout.py      #   torchao Layout 集成
│   ├── setup_cuda.py              #   CUDA 编译脚本
│   └── hifloat8_cuda/             #   HiFloat8 CUDA 内核源码
│       ├── hifloat8_encode_decode.cu  # encode/decode + fake_quant (CUDA/CPU)
│       ├── hifloat8_lut.h         #   127 值查找表
│       ├── hif8_round.cuh         #   CUDA 舍入函数
│       └── hif8_round_cpu.h       #   CPU 舍入函数 (float + double)
├── quantization/                  # Layer 2: 量化模块
│   ├── hifp8_config.py            #   配置类 (BF16/uint8/KV cache)
│   ├── hifp8_fake_quantizer.py    #   伪量化器（支持运行时 kernel 替换）
│   └── hifp8_linear.py            #   HiFP8FakeQuantizedLinear
├── export/                        # Layer 3: 导出
│   ├── bf16_export.py             #   统一导出入口 (bf16/uint8)
│   ├── uint8_export.py            #   uint8 真量化导出
│   ├── hif8_export.py             #   HiF8 预量化导出 (vLLM-HiF8 fork)
│   └── vllm_export.py             #   Float8Tensor 导出
├── vllm_plugin/                   # Layer 4: vLLM 集成
│   ├── hifp8_loader.py            #   双模式加载器（自动格式检测）
│   ├── hifp8_uint8_linear.py      #   uint8 Linear 层（eager/lazy 解码）
│   ├── hifp8_vllm_patcher.py      #   vLLM 0.12.0 架构感知 patcher
│   └── hifp8_kv_cache_patcher.py  #   KV cache 量化 patcher
├── scripts/                       # 工具脚本
│   ├── start_vllm_hifp8_server_v4.py  # vLLM server（双模式）
│   ├── eval_arc_comparison.py         # ARC 评测对比脚本
│   ├── eval_hif8_vllm.py             # HiF8 端到端评测脚本
│   ├── compare_gpt_oss_outputs.py    # GPT-OSS BF16 vs HiF8 输出对比
│   └── generate_lut.py                # HiFloat8 LUT 生成
├── tests/                         # 测试
│   ├── test_hifp8_flow.py         #   核心测试 (含 CPU/double/direct fake_quant)
│   ├── test_hifp8_uint8_layout.py #   uint8 layout 测试
│   ├── test_hifp8_kv_cache.py     #   KV cache 测试
│   └── test_smooth_hif8_export.py #   SmoothQuant + HiF8 导出测试
├── examples/                      # 示例
│   ├── quantize_model.py
│   └── quantize_qwen3.py
└── ao/                            # torchao 源码（只读）
```

## 快速开始

### 环境准备

```bash
# 设置 PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/ao:$PYTHONPATH"

# 编译 HiFloat8 CUDA 内核（uint8 真量化、direct fake_quant 需要）
cd custom_ops && python setup_cuda.py build_ext --inplace && cd ..
```

**Requirements**: Python >= 3.10, PyTorch >= 2.0 (CUDA), safetensors, transformers
**Optional (deployment)**: vLLM 0.12.0 (standard or [HiF8 fork](https://github.com/XiangWanggithub/vllm.git)), evalscope

### 模式 1：BF16 伪量化

训练/校准阶段使用，在 BF16 精度下模拟量化误差。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization.hifp8_linear import prepare_hifp8_fake_quant
from export.bf16_export import export_for_vllm

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                              torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 2. 应用伪量化
prepare_hifp8_fake_quant(model)

# 3. (可选) 校准 / 微调 ...

# 4. 导出 BF16 格式
export_for_vllm(model, tokenizer, "./output/qwen3_bf16", export_mode="bf16")
```

**导出产物**：
```
output/qwen3_bf16/
├── model.safetensors      # BF16 权重 + scale buffers
├── config.json
├── hifp8_metadata.json    # 量化元数据
└── tokenizer files
```

### 模式 2：uint8 真量化

部署阶段使用，将权重编码为 HiFloat8 uint8 格式，实现 **2x 压缩**。

```python
# 沿用上面已伪量化的模型，导出 uint8 格式
export_for_vllm(model, tokenizer, "./output/qwen3_uint8", export_mode="uint8")
```

**导出产物**：
```
output/qwen3_uint8/
├── model.safetensors      # uint8 编码权重 + FP32 scales
├── config.json
├── hifp8_metadata.json    # weight_format: "uint8_hifloat8"
└── tokenizer files
```

### 模式 3：HiF8 预量化导出

为 vLLM-HiF8 fork 导出预量化权重（BF16 fake-quantized + per-channel scale），支持 torch.compile 加速。

```python
from export.hif8_export import export_for_hif8_vllm

export_for_hif8_vllm(model, tokenizer, "./output/gpt_oss_hif8",
                      per_channel=True, activation_scheme="dynamic")
```

**导出产物**：
```
output/gpt_oss_hif8/
├── model.safetensors      # BF16 fake-quantized 权重 + FP32 per-channel scales
│                          # + smooth_scale (如启用 SmoothQuant)
├── config.json            # quantization_config: {"quant_method": "hif8",
│                          #   "has_smooth_scale": true/false, ...}
└── tokenizer files
```

**SmoothQuant 支持**：若模型经过 `apply_smooth_scale()` 处理，导出时自动包含 `{layer}.smooth_scale` 张量，vLLM-HiF8 fork 运行时在量化前对 activation 执行 `x / smooth_scale`。

> 搭配下文 [路径 A：vLLM-HiF8 fork](#路径-avllm-hif8-fork推荐支持-torchcompile) 部署推理。

### vLLM 部署

本项目提供两种 vLLM 部署路径：

#### 路径 A：vLLM-HiF8 fork（推荐，支持 torch.compile）

基于同事修改的 vLLM fork，原生支持 HiFloat8 量化推理，支持 torch.compile 加速。

**项目地址**：https://github.com/XiangWanggithub/vllm.git （分支：`v0.12.0`）

**安装**：
```bash
git clone -b v0.12.0 https://github.com/XiangWanggithub/vllm.git vllm-hifp8
cd vllm-hifp8
VLLM_USE_PRECOMPILED=1 pip install -e .
```

**工作原理**：vLLM-HiF8 fork 在 `config.json` 中识别 `quant_method: "hif8"`，加载预量化的 BF16 权重和 per-channel `weight_scale`，运行时仅对 activation 做 fake quant，支持 torch.compile 图模式加速。当 `has_smooth_scale=true` 时，运行时在量化前自动应用 `x / smooth_scale`。

**搭配导出模式 3（HiF8 预量化导出）使用**：
```bash
# 启动 server（torch.compile + 禁用 CUDA graph）
python -m vllm.entrypoints.openai.api_server \
    --model ./output/gpt_oss_hif8 \
    --served-model-name gpt-oss-hif8 \
    --tensor-parallel-size 2 \
    --compilation-config '{"cudagraph_mode": 0}' \
    --gpu-memory-utilization 0.95 \
    --port 8000
```

> 注意：HiF8 的 fake_quant kernel 内部分配内存，不兼容 CUDA graph capture，需设置 `cudagraph_mode: 0`。这保持 torch.compile 优化但跳过 CUDA graph。

#### 路径 B：vLLM 插件（monkey-patching，适用于标准 vLLM）

通过本项目的 `vllm_plugin/` 模块，以 monkey-patching 方式将 HiFP8 量化集成到标准 vLLM 0.12.0 中，无需修改 vLLM 源码。

**搭配导出模式 1（BF16）或模式 2（uint8）使用**：

`vllm_plugin/` 模块说明：

| 文件 | 功能 |
|------|------|
| `hifp8_loader.py` | 双模式加载器，自动检测 BF16/uint8 格式 |
| `hifp8_vllm_patcher.py` | vLLM 0.12.0 架构感知 patcher，patch QKVParallelLinear/RowParallelLinear/ColumnParallelLinear |
| `hifp8_uint8_linear.py` | uint8 Linear 层，支持 eager（加载时解码）和 lazy（推理时解码）两种策略 |
| `hifp8_kv_cache_patcher.py` | KV cache 量化 patcher |

```bash
# 使用 v4 server（自动检测量化格式：BF16 / uint8 / 无量化）
python scripts/start_vllm_hifp8_server_v4.py \
    --model ./output/qwen3_bf16 \
    --port 8000 \
    --served-model-name qwen3-hifp8
```

v4 server 工作流程：
1. Hook vLLM 的 `DefaultModelLoader.load_model()`
2. 读取 `hifp8_metadata.json` 自动检测格式
3. BF16 格式 → patch forward pass 注入 fake quant
4. uint8 格式 → 加载时解码 uint8 权重回 BF16
5. 无量化 → 直接透传，不做修改

#### 两种路径对比

| | 路径 A：vLLM-HiF8 fork | 路径 B：vLLM 插件 |
|---|---|---|
| vLLM 版本 | [XiangWanggithub/vllm](https://github.com/XiangWanggithub/vllm.git) v0.12.0 | 标准 vLLM 0.12.0 |
| 导出格式 | 模式 3（HiF8 预量化） | 模式 1（BF16）/ 模式 2（uint8） |
| torch.compile | 支持 | 不支持 |
| 集成方式 | vLLM 原生 `quant_method` | monkey-patching |
| 推荐场景 | 生产部署、大模型推理 | 快速验证、标准 vLLM 环境 |

**测试推理**：
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-hifp8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Evalscope 评测

```bash
# 一键评测：导出 + 启动 server + 评测 ARC (original vs bf16 vs uint8)
python scripts/eval_arc_comparison.py \
    --model /home/models/Qwen3-0.6B \
    --gpu 0 \
    --limit 100

# HiF8 端到端评测 (导出 + vLLM server + ARC)
python scripts/eval_hif8_vllm.py \
    --model /home/models/gpt-oss-20b-BF16 \
    --output /home/data/hifp8_eval/gpt_oss_20b_hif8 \
    --tp 2

# GPT-OSS BF16 vs HiF8 输出对比
python scripts/compare_gpt_oss_outputs.py \
    --bf16-model /home/models/gpt-oss-20b-BF16 \
    --hif8-model /home/data/hifp8_eval/gpt_oss_20b_hif8 \
    --tp 2

# 或手动评测
evalscope eval \
    --model qwen3-hifp8 \
    --api-url http://localhost:8000/v1 \
    --datasets arc \
    --dataset-hub modelscope
```

### 运行测试

```bash
# 全部 81 个测试
python -m unittest tests.test_hifp8_flow tests.test_hifp8_uint8_layout tests.test_hifp8_kv_cache tests.test_smooth_hif8_export -v
```

## 技术架构

### 量化流程

```
                    量化/导出                                 部署

                  ┌──────────────────┐    模式 1    ┌──────────────────────┐
                  │ BF16 伪量化       │───────────→  │ vLLM 插件 (路径 B)    │
                  │ float→hif8→float │              │ 运行时 fake quant     │
                  │ (CUDA kernel)    │              └──────────────────────┘
                  └──────────────────┘
原始模型 (BF16) ─→                                  ┌──────────────────────┐
                  ┌──────────────────┐    模式 2    │ vLLM 插件 (路径 B)    │
                  │ uint8 编码        │───────────→  │ 加载时解码回 BF16     │
                  │ float→uint8      │              └──────────────────────┘
                  │ (HiFloat8 LUT)   │
                  └──────────────────┘              ┌──────────────────────┐
                                          模式 3    │ vLLM-HiF8 fork(路径A)│
                  ┌──────────────────┐              │ 预量化权重            │
                  │ HiF8 预量化导出   │───────────→  │ + torch.compile      │
                  │ fake_quant→BF16  │              └──────────────────────┘
                  │ + per-ch scale   │
                  └──────────────────┘
```

### HiFloat8 编码格式

```
8-bit = Sign(1) | Index(7)

127 个正值 + 127 个负值 + 0 + NaN = 256 个编码
自适应精度：小指数多精度，大指数少精度
Per-row scaling：每行独立 FP32 scale
Inf 编码：0x7F (index=127)
```

### Kernel 架构与 NPU 适配

当前实现已包含完整的 HiFloat8 CUDA kernel（`custom_ops/hifloat8_cuda/`），支持 float32/float64/bfloat16 及 CPU fallback。

当 NPU kernel 可用时，只需修改一个函数即可完成替换：

```python
# custom_ops/hifp8_ops.py - 唯一需要修改的文件
def hifp8_fake_quantize(x, param1=0, param2=0, *, granularity, target_dtype):
    scale = compute_hifp8_scale(x, param1, param2, granularity)
    q = npu_hifp8_quantize(x, scale, param1, param2)     # ← NPU kernel
    dq = npu_hifp8_dequantize(q, scale, original_dtype)
    return dq
```

或通过运行时替换（适用于 CUDA/NPU A/B 对比测试）：

```python
for module in model.modules():
    if isinstance(module, HiFP8FakeQuantizer):
        module.set_quantize_fn(my_npu_kernel)
```

当前 CUDA kernel 可直接调用：

```python
from custom_ops.hifp8_uint8_ops import hifp8_fake_quant_direct

output = hifp8_fake_quant_direct(input_tensor)  # 自动选择 CUDA 或 CPU 路径
```

## KV Cache 量化

支持 KV cache HiFP8 量化，节省 ~50% KV cache 内存：

```python
from quantization.hifp8_config import HiFP8KVCacheConfig, QuantMode

kv_config = HiFP8KVCacheConfig(
    enabled=True,
    mode=QuantMode.STATIC,        # STATIC (推理) 或 DYNAMIC (校准)
)

export_for_vllm(model, tokenizer, output_dir, kv_cache_config=kv_config)
```

## API 参考

### 核心函数

| 函数 | 说明 |
|------|------|
| `prepare_hifp8_fake_quant(model)` | 将 nn.Linear 替换为 HiFP8FakeQuantizedLinear |
| `export_for_vllm(model, tokenizer, dir, export_mode)` | 统一导出 (bf16/uint8) |
| `export_for_hif8_vllm(model, tokenizer, dir)` | HiF8 预量化导出 (vLLM-HiF8 fork) |
| `apply_hifp8_to_vllm_model(model, dir)` | 自动检测格式并加载量化权重 |
| `hifp8_fake_quantize(x, p1, p2, *, granularity, dtype)` | 核心伪量化（kernel 替换点） |
| `hifp8_fake_quant_direct(x)` | C++ kernel 直接 fake quant (CUDA/CPU) |
| `apply_smooth_scale(model, scales)` | 应用 SmoothQuant scale 到模型 |

### 配置类

| 类 | 说明 |
|----|------|
| `HiFP8FakeQuantizeConfig` | 单张量伪量化配置 (granularity, dtype, mode) |
| `HiFP8QuantizationConfig` | 顶层配置，兼容 torchao `quantize_()` API |
| `HiFP8KVCacheConfig` | KV cache 量化配置 |

### vLLM Server

```bash
python scripts/start_vllm_hifp8_server_v4.py --model <path> --port 8000
```

API 端点：
- `GET  /health` — 健康检查
- `GET  /v1/models` — 模型列表
- `POST /v1/chat/completions` — 聊天补全 (OpenAI 兼容)
- `POST /v1/completions` — 文本补全

## 故障排除

### CUDA 版本不匹配

```
RuntimeError: The detected CUDA version (13.0) mismatches PyTorch (12.8)
```

`setup_cuda.py` 已内置版本检查 bypass，通常自动解决。若仍有问题：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### 找不到 CUDA 内核

```
Warning: hifp8_cuda_uint8 CUDA extension not found
```

```bash
cd custom_ops && python setup_cuda.py build_ext --inplace
```

### Import 错误

```bash
export PYTHONPATH="/path/to/hifp8:/path/to/hifp8/ao:$PYTHONPATH"
```

## TODO

- [ ] **SmoothQuant scale fusion**：将 `1/smooth_scale` 融合到前置层权重中，消除运行时除法开销
  - `q_proj/k_proj/v_proj` ← 融合到 `input_layernorm.weight`
  - `o_proj` ← 融合到 `v_proj` 权重
  - `gate_proj/up_proj` ← 融合到 `post_attention_layernorm.weight`
  - `down_proj` ← 融合到 `up_proj` 权重
- [ ] **vLLM-HiF8 fork 远程推送**：本地已提交 (747f17fe6)，需解决 HTTPS 认证后推送
- [ ] **更多模型验证**：在更多模型上验证 SmoothQuant + HiF8 端到端精度

## 参考

- **HiFloat8 论文**: [arxiv 2409.16626](https://arxiv.org/abs/2409.16626)
- **vLLM-HiF8 fork**: [XiangWanggithub/vllm](https://github.com/XiangWanggithub/vllm.git) (分支 `v0.12.0`，原生支持 HiFloat8)
- **torchao**: `./ao/` (v0.14.1, 只读)
- **vLLM**: 0.12.0

## License

BSD-3-Clause
