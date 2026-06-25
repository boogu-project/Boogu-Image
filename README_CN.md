<p align="center">
  <img src="assets/boogu-logo-title.svg" alt="Boogu-Image-0.1" width="420" />
</p>

<h3 align="center">助推开源统一多模态理解与生成</h3>

<div align="center">

<img src="assets/boogu-infinity-teaser.png" alt="Boogu-Image-0.1 Teaser" width="100%" />



<!-- ============== Badges ============== -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-{{ paper_id }}-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/{{ paper_id }}) -->
[![项目主页](https://img.shields.io/badge/🌐-Project%20Page-blue)](https://boogu.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/Boogu)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white)](https://github.com/boogu-project/Boogu-Image)
[![ModelScope](https://img.shields.io/badge/🤖-ModelScope-624aff)](https://modelscope.cn/organization/Boogu)

[![Demo-Base](https://img.shields.io/badge/🎨-Demo%20Base-ff69b4)](http://demo-base.boogu.org/)
[![Demo-Edit](https://img.shields.io/badge/🖌️-Demo%20Edit-ff8c00)](http://demo-edit.boogu.org/)
[![Demo-Turbo](https://img.shields.io/badge/⚡-Demo%20Turbo-9b59b6)](http://demo-turbo.boogu.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![技术报告](https://img.shields.io/badge/📄-Technical%20Report%20(Coming%20Soon)-lightgrey)]()


欢迎来到 **Boogu-Image-0.1** 官方仓库！

[English](./README.md) | 中文

</div>

---

> ## ⚠️ 重要声明
>
> **Boogu 团队目前未推出任何针对 Boogu-Image 的收费 API、订阅或商业化服务。** 任何以 **"Boogu-Image"** 或其相近、变体名称（如 `booguimage`、`Boogu Image`、`Boogu` 等）名义提供的收费产品或服务，**均与本项目无关**，属于非官方行为。请在付款前务必仔细甄别，谨防上当受骗，切实保护好个人隐私与财产安全。
>
> **Boogu-Image-0.1 仅为一个研究项目，并非官方模型发布。**

## 📖 简介

**Boogu-Image-0.1** 是一个在当前开源模型中具备强竞争力的 **Apache-2.0 开源统一图像生成与编辑模型家族**，包含 **Base**、**Turbo**、**Edit** 等多个变体，在高质量文生图、快速生成、图像编辑和中英文文字渲染等任务上提供稳定实用的能力。像 Nano Banana Pro 和 GPT-Image-2 这样的闭源多模态理解与生成系统之所以表现卓越，并非依赖单一模型，而是得益于一整套高度统一的系统能力。然而，在训练资源相较于闭源模型极端有限的条件下，我们发现，通过系统性地增强模型的理解能力、数据质量和训练流程，仍然能够显著提升图像生成与编辑的性能。具体而言，相比其他已知的开源工作，我们使用的训练数据规模大约少一个数量级。我们希望这项经验研究与开源发布，能够推动多模态生成与理解领域的开源生态发展。

本仓库提供 **Boogu-Image-0.1** 的模型权重和推理代码。

## 🏆 Boogu Arena

由于我们无法直接在 LM Arena 上评测，我们构建了 **Boogu Arena**——一套 LM Arena 风格的偏好评测。我们使用 LLM 生成多样化的用户画像（persona），再让每个画像产出图像生成提示词，共得到 **1K+ 条测试提示词**，并将公开发布以供社区复现。下方的 ELO 排行榜涵盖了领先的闭源与开源系统。我们欢迎对结果有疑问的团队与我们联系，以便我们能够努力实现更加客观、公平和可重复的评估。

<!-- <p align="center">
  <img src="assets/ci_chart.svg" alt="Boogu Arena ELO 排行榜" width="100%" />
</p> -->
<p align="center">
  <img src="assets/boogu_elo_arena.svg" alt="Boogu Arena ELO Leaderboard" width="100%" />
</p>

## ✨ 亮点

- 📸 **精准优美的摄影** — 准确理解摄影类提示词，生成具有自然光照、连贯构图和真实细节的高质量图像；即使在更复杂的真实场景中，也能保持主体、背景与空间关系的一致性
![Showcase for Photography](assets/photography_triptych.png)
- 📝 **多样稳定的文字渲染** — 支持海报、邮票、文档、界面、品牌规范、手写板等多种文字密集型设计，在多样布局下实现可读的结构、稳定的排版以及稳健的中英文双语渲染
![Showcase for Photography](assets/text_rendering_triptych.png)
- 🎨 **多样精美的风格化** — 涵盖微缩 3D 场景、国风鎏金美学、闪耀奇幻视觉、动漫肖像、神话角色等风格化生成；不只是风格迁移，更追求稳定、美观、贴合提示词的创意生成
![Showcase for Photography](assets/stylization_triptych.png)
- 🖌️ **多样化图像编辑** — 支持丰富的编辑任务，包括物体的插入、替换与移除，属性与材质修改，背景与场景替换，以及在多种艺术风格之间的忠实风格迁移，同时保持原始主体与构图的一致性
![Showcase for Image Editing](assets/edit_cases_combined.png)
![Showcase for Style Transfer](assets/style_cases_combined.png)
- 🪧 **个性化海报设计与产品图渲染** — 生成个性化的海报版式与简洁的产品图，具备一致的品牌调性、精致的排版以及产品级的光照与构图
![Showcase for Poster & Product](assets/poster_cases_combined.png)
- ✍️ **精准文字编辑** — 支持图像内细粒度的文字编辑——在中英文中替换、新增或删除字符——并可灵活适配不同的字体、字重、颜色与版式，匹配多样化的设计意图
![Showcase for Text Editing](assets/text_cases_combined.png)
- 📊 **极具竞争力的综合性能** — 在众多场景和基准上展现出极具竞争力的性能，Boogu-Image-0.1 家族在 Boogu Arena 中位居参评开源与闭源系统的前列

> 📖 完整的实践经验与对当前局限性的坦诚说明，请参阅下文的 [安全性与局限性](#安全性与局限性)。

## 🔬 分场景对比

在整体 Arena 排名之外，我们将性能按场景拆分，与主流开源模型进行对比。评分基于我们对各类典型提示词的内部评测。

| 模型 | 写实摄影 | 简单文字渲染 | 密集文字渲染 |
| :--- | :---: | :---: | :---: |
| **Boogu-Image-0.1-Turbo** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Boogu-Image-0.1-Base**  | ⭐⭐⭐  | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Z-Image-Turbo             | ⭐⭐⭐⭐ | ⭐⭐⭐  | ⭐⭐ |
| Qwen-Image-2512           | ⭐⭐⭐  | ⭐⭐⭐⭐ | ⭐⭐⭐ |

- 📸 **兼顾写实摄影与稳定文字渲染** — Boogu-Image-0.1-Turbo 能够生成写实摄影图像，同时在简单与密集文字渲染上也都有稳健的表现。
- 📝 **密集文字渲染表现稳健** — Boogu-Image-0.1-Base 在密集、复杂排版的文字场景（海报、文档、品牌规范、复杂中英文设计等）中展现出有竞争力的表现。
- 💡 **使用建议** — 当工作负载以密集 / 超密集文字渲染为主时，推荐使用 **Boogu-Image-0.1-Base 并以 2K 分辨率输出**，以获得最佳的排版忠实度和文字准确度。

## 📣 最新动态
- **2026-06-17** 🔥 [**ComfyUI-Boogu**](https://huggingface.co/Comfy-Org/Boogu-Image) 由 ComfyUI 提供支持的版本已发布！感谢 ComfyUI！
- **2026-06-17** 🔥 [**ComfyUI-Boogu**](https://github.com/boogu-project/ComfyUI-Boogu) 发布! 
- **2026-06-16** 🔥 **Boogu-Image-0.1-Base（文生图）发布！** 核心文生图基础模型。体验[在线演示](http://demo-base.boogu.org/)。
- **2026-06-16** 🎨 **Boogu-Image-0.1-Edit（图生图）发布！** 图像编辑和转换能力现已可用。体验[在线演示](http://demo-edit.boogu.org/)。
- **2026-06-16** 🚀 **Boogu-Image-0.1-Turbo 发布！** 用于快速推理与照片级真实感生成的 4 步蒸馏变体。体验[在线演示](http://demo-turbo.boogu.org/)。
<!-- - **[{{ 2026-06-DD }}]** 📄 **技术报告发布！** 阅读我们在 [arXiv](https://arxiv.org/abs/{{ paper_id }}) 上的发现。 -->

## 📥 模型库

| 模型 | 参数量 | 训练方式 | 步数 | CFG | 任务 | Hugging Face | 魔塔社区 | 演示 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Boogu-Image-0.1-Base** | 10B | 联合训练 | 25~50 | 2.0～5.0<br>（例如 4.0） | 文生图 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-yellow)](https://huggingface.co/Boogu/Boogu-Image-0.1-Base) | [![MS](https://img.shields.io/badge/🤖-Checkpoint-624aff)](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Base) | [![Demo](https://img.shields.io/badge/🎨-Demo-ff69b4)](http://demo-base.boogu.org/) |
| **Boogu-Image-0.1-Base-fp8** | 10B | 联合训练| 25~50 | 2.0～5.0<br>（例如 4.0） | 文生图 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-yellow)](https://huggingface.co/Boogu/Boogu-Image-0.1-Base-fp8) | [![MS](https://img.shields.io/badge/🤖-Checkpoint-624aff)](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Base-fp8) | — |
| **Boogu-Image-0.1-Edit** | 10B | 联合训练 | 25~50 | 2.0～5.0<br>（例如 5.0） | 图生图编辑 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-yellow)](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit) | [![MS](https://img.shields.io/badge/🤖-Checkpoint-624aff)](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Edit) | [![Demo](https://img.shields.io/badge/🖌️-Demo-ff8c00)](http://demo-edit.boogu.org/) |
| **Boogu-Image-0.1-Edit-fp8** | 10B | 联合训练 | 25~50 | 2.0～5.0<br>（例如 5.0） | 图生图编辑 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-yellow)](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit-fp8) | [![MS](https://img.shields.io/badge/🤖-Checkpoint-624aff)](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Edit-fp8) | — |
| **Boogu-Image-0.1-Turbo** | 10B | + 解耦 DMD | 4 | 1.0 | 文生图 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-yellow)](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo) | [![MS](https://img.shields.io/badge/🤖-Checkpoint-624aff)](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Turbo) | [![Demo](https://img.shields.io/badge/⚡-Demo-9b59b6)](http://demo-turbo.boogu.org/) |
| **Boogu-Image-0.1-Turbo-fp8** | 10B | + 解耦 DMD | 4 | 1.0 | 文生图 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-yellow)](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo-fp8) | [![MS](https://img.shields.io/badge/🤖-Checkpoint-624aff)](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Turbo-fp8) | — |

- **Boogu-Image-0.1-Base**：基础模型，具备强**多样性**与**可控性**——适合**微调**及下游开发。主要面向**超密集文字渲染**等复杂多文本场景；若追求照片级真实感，Turbo 模型通常是更好的选择。支持分辨率：1K、1.5K、2K。
- **Boogu-Image-0.1-Edit**：图像编辑与转换变体。支持分辨率：1K、1.5K、2K。
- **Boogu-Image-0.1-Turbo**：蒸馏变体，与基础模型**参数量相同**，通常仅需 **3~4 步**。专注于**高质量生成**与照片级真实感，同时保留双语文字渲染与提示词遵循能力。支持分辨率：1K。

如无特别说明，所有模型变体均支持以下宽高比：1:1、2:3、3:2、3:4、4:3、1:2、2:1、9:16、16:9。

## 🛠️ 安装

> **测试环境：** Python 3.10 · CUDA 12.6 · PyTorch 2.7.1

```bash
# 使用全新的 conda 环境
conda create -y -n boogu python=3.10
conda activate boogu
# 安装必要的依赖
# 支持 PyTorch 最高 2.11.0，CUDA 最高 12.8
# 查看 `requirements/<torch>_<cuda>.txt`
pip install -r requirements/torch2.7-cu126.txt
pip install -e .
python utils/get_flash_attn.py
```

或者

```bash
bash quick_start.sh
conda activate boogu
```

### 下载模型权重
在运行推理之前，请将模型权重下载到本地 `models/` 目录。我们推荐使用 Hugging Face 官方 CLI：

```bash
pip install -U "huggingface_hub[cli]"

# 下载到 ./models/<model-name>
huggingface-cli download Boogu/Boogu-Image-0.1-Base --local-dir models/Boogu-Image-0.1-Base
huggingface-cli download Boogu/Boogu-Image-0.1-Turbo --local-dir models/Boogu-Image-0.1-Turbo
huggingface-cli download Boogu/Boogu-Image-0.1-Edit --local-dir models/Boogu-Image-0.1-Edit
```



下载后的目录结构示例：

```
models/
└── Boogu-Image-0.1-Base/
    ├── model_index.json
    ├── mllm
    ├── processor
    ├── scheduler
    ├── transformer
    └── vae
```

然后通过 `--model models/Boogu-Image-0.1-Base` 指向本地路径进行推理。

### Flash Attention

本仓库提供 `utils/get_flash_attn.py` 来自动安装适配您环境的 `flash-attn` wheel。

环境要求：
- 已安装 Python 和带 CUDA 的 PyTorch
- Linux x86_64

```bash
# 自动模式：检测环境，下载预编译 wheel，回退到源码编译
python utils/get_flash_attn.py

# 强制源码编译
python utils/get_flash_attn.py --build
```

该脚本首先搜索 [`mjun0812/flash-attention-prebuild-wheels`](https://github.com/mjun0812/flash-attention-prebuild-wheels)，然后尝试官方 [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) 发布的 wheel（包含两种 cxx11abi 变体），最后回退到通过 `pip install flash-attn --no-build-isolation` 进行源码编译。


## 🚀 快速开始

### PyTorch 原生文生图推理

```bash
export device="cuda:0" # 必需

# Prompt 增强由 instruction reasoner（也称 rewriter）提供。
# 我们提供两种使用方式：
#
# 1. 外挂式 standalone rewriter：
#    可参考 utils/t2i_external_prompt_rewriter.py。这是一个纯外挂模式示例，
#    需要充足的 GPU 显存，且没有复杂的显存管理机制。
#    python utils/t2i_external_prompt_rewriter.py --prompt "画一只猫" --model /path/to/Qwen3-VL-32B-Instruct --lang zh
#
# 2. Pipeline 内置 rewriter：
#    可参考 demo_scripts 下名称包含 "reasoning" 的脚本。
#    例如：demo_scripts/demo_t2i_local_reasoning.sh
#    这种方式支持更灵活的显存管理。请手动设置生成模型和 rewriter 的设备，
#    然后传给 inference.py：
#    export device="cuda:0"
#    export rewriter_device="cuda:1"
#    python inference.py --device $device --rewriter_device $rewriter_device ...
#    更多详情请参考 INFERENCE_GUIDE.md。

python inference.py \
  --pretrained_pipeline_name_or_path "models/Boogu-Image-0.1-Base" \
  --instruction "一幅国风琉金风格的山水画作，展现了桂林山水在金光普照下的壮丽景象。远山层叠，江水如镜，山峰边缘勾勒着发光的金色线条。画面采用石青石绿岩彩与鎏金质感相结合，局部有厚涂油画笔触，空中飘浮着金色粒子，营造出梦幻朦胧而又磅礴大气的意境。" \
  --num_inference_steps 50 \
  --height 1024 --width 1024 \
  --text_guidance_scale 4.0 \
  --output_image_path "outputs/test_base/out_1.png" \
  --device "$device"
```

### 硬件说明

> 📖 完整的命令行选项、设备设置、卸载策略、缓存加速、Torch Compile、FP8 和批量推理详情，请参阅 [**INFERENCE_GUIDE.md**](./INFERENCE_GUIDE.md)。
> Torch Compile 注意事项：`--enable_torch_compile` 在某些 GPU/模型上偶尔会产生全黑输出。如遇此情况，请先禁用该选项。

| 显存 | 推荐配置（文生图 1K）                                                                                           | 推荐配置（文生图 2K）                                                                                           |
|------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 12GB | 未量化：`--enable_sequential_cpu_offload_flag`<br>量化：`--enable_model_cpu_offload_flag --use_fp8_weights` | 未量化：`--enable_sequential_cpu_offload_flag`<br>量化：`--enable_group_offload_flag --use_fp8_weights`     |
| 16GB | 未量化：`--enable_sequential_cpu_offload_flag`<br>量化：`--enable_model_cpu_offload_flag --use_fp8_weights` | 未量化：`--enable_sequential_cpu_offload_flag`<br>量化：`--enable_model_cpu_offload_flag --use_fp8_weights` |
| 24GB | 未量化：`--enable_model_cpu_offload_flag`<br>量化：`--use_fp8_weights`                                       | `--enable_model_cpu_offload_flag`                                                                                     |
| 32GB | 未量化：`--enable_model_cpu_offload_flag`<br>量化：`--use_fp8_weights`                                      | 未量化：`--enable_model_cpu_offload_flag`<br>量化：`--use_fp8_weights`                                      |
| 40GB | 基础模型                                                                                                            | 未量化：`--enable_model_cpu_offload_flag`<br>量化：`--use_fp8_weights`                                      |
| 80GB | 基础模型                                                                                                            | 基础模型                                                                                                            |

## ⚠️ 安全性与局限性

### 安全性

**Boogu-Image-0.1** 以**研究目的**发布，未经额外安全措施不建议用于生产环境部署。我们在数据筛选、训练和评估过程中考虑了负责任的 AI 因素，但模型仍可能产生不准确、有偏见或不适当的输出。

### 已知局限性

**🌍 世界知识差距**
- 对于需要丰富常识、领域知识、真实品牌或人物、著名地标、名人、产品或复杂上下文理解的任务，Boogu 与强大的闭源系统仍有明显差距
- 这一能力的评测代价极高，即使 Arena 风格的评测也难以完整衡量，因此现有基准几乎无法量化这一维度，真实差距很可能比测得的分数更大

**🖼️ 图生图一致性与上下文场景**
- 对于需要严格保持输入主体、身份、布局或精细细节的编辑任务，Boogu 的图生图一致性尚不够稳定
- 我们的图生图能力更侧重摄影与文字等应用场景，因此在部分上下文生成（in-context）场景中，Boogu 仍落后于 **Seedream 5.0** 和 **Nano Banana Pro**

**📝 文字渲染稳定性**
- Boogu 可以处理许多中文和英文文字场景，但长文本、密集排版、小字号以及复杂设计布局仍可能产生错别字、缺字或布局漂移
- 文字渲染目前主要面向中文和英文；其他语言没有专门优化，效果可能明显退化

**🦴 复杂姿势下的身体结构**
- 在多人互动、遮挡、夸张动作或不寻常视角下，手部、肢体和身体结构仍可能变得不自然或不一致

**👤 小尺寸人脸与小肢体**
- 由于我们使用开源的 **FLUX.1 VAE**，重建损失相对较大，因此小人脸、小肢体、眼睛和文字等细节仍可能出现伪影或不稳定

**📦 开源范围有限**
- 受资源限制、工程复杂度和发布边界的约束，我们无法开源全部训练与系统细节
- 本次发布在可复现性、可用性与可持续维护之间取得平衡，为社区研究提供一个可靠的起点

下游用户有责任根据其使用场景应用适当的内容审核、验证和合规检查。


## 🙏 致谢

[GPT-Image](https://openai.com/index/introducing-chatgpt-images-2-0/)、[Nano Banana](https://gemini.google/overview/image-generation/) 以及 [Seedream](https://seed.bytedance.com/en/seedream5_0_lite) 系列等闭源系统帮助我们更好地理解统一理解-生成系统的前沿能力与实际边界。我们感谢 [Qwen-Image](https://github.com/QwenLM/Qwen-Image)、[Z-Image](https://github.com/Tongyi-MAI/Z-Image)、[OmniGen2](https://github.com/VectorSpaceLab/OmniGen2)、[FLUX](https://github.com/black-forest-labs/flux) 以及更广泛的开源社区所提供的宝贵基础与参考，也感谢 [DeepSeek](https://www.deepseek.com) 提供了足够强大的开源理解模型，为开源统一多模态理解-生成系统的发展提供了重要支持。


## 📄 许可证

本项目基于 [Apache-2.0 许可证](LICENSE) 发布。
