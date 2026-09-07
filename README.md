# Krea 2 RGBA VAE：Linux / Accelerate / DeepSpeed ZeRO-2

训练 `RGBA → encoder → latent → decoder → RGBA`，支持 **AVIF、RGB/RGBA 混合训练**。兼容性训练通过冻结的原 RGB VAE，约束新 VAE 保留原 latent 的 RGB 解码能力。所有新增训练项均可关闭；训练脚本不加载 DiT 或文本编码器，不需要 caption。

## 目标与边界

你需要的是两阶段路线：

1. 训练 RGBA VAE，同时尽量保留原 Krea 2 DiT 的 latent 含义；原 DiT 不训练时，读取新 decoder 的 RGB 三通道输出。
2. 固定训练好的 RGBA VAE，再用带 caption 的 RGBA 数据适配 DiT，例如 LoRA。此时 DiT 学习产生包含透明度信息的 latent。

本项目实现第一阶段及 latent 接口工具。**保持 latent 形状相同，不等于语义兼容；约束编码器，也不等于解码器不会漂移。** 所以提供编码分布对齐和旧 latent 解码保持两类约束。
这些是软约束，没有保证原 DiT 画质完全不变，也没有保证后续某个固定的少量样本数就能学会 RGBA。必须在固定 prompt/seed 的真实 DiT 输出上比较新旧解码结果。

## 模型来源与环境

默认使用官方 `krea/Krea-2-Turbo` 的 `vae/`，revision 固定为 `98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7`。配置为 `AutoencoderKLQwenImage`，原 RGB 输入/输出 3 通道，latent 16 通道，空间压缩 8 倍。

- [官方 VAE 配置](https://huggingface.co/krea/Krea-2-Turbo/blob/main/vae/config.json)
- [Krea 2 官方仓库](https://github.com/krea-ai/krea-2)
- [Diffusers VAE 实现](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py)

HF 仓库需要访问授权：先在模型页接受条款，然后执行 `hf auth login`。脚本只加载 VAE。原模型和教师必须对应你实际使用的 DiT；相同 latent 通道数并不足以让不同模型互换。

目标环境为已有的 **Linux + CUDA + Accelerate + DeepSpeed ZeRO-2**，Python 3.10–3.12。沿用匹配的 PyTorch、torchvision 和 DeepSpeed，依赖文件不安装 DeepSpeed：

```bash
pip install -r requirements.txt
```

AVIF 使用 Pillow 的原生解码器，要求 **Pillow >=11.3 且构建包含 AVIF 支持**；官方 Linux wheel 包含该功能。[Pillow 发布说明](https://pillow.readthedocs.io/en/stable/releasenotes/11.3.0.html)
LPIPS 首次运行会下载 torchvision AlexNet 权重；离线需预缓存，或用 `--lpips-weight 0` 关闭。

## AVIF 与 RGB/RGBA 混合数据

```text
/data/mixed/train/
  cutout.png
  transparent.avif
  photo.avif
  landscape.jpg
/data/mixed/val/
  ...
```

递归读取 `.avif/.png/.webp/.tif/.tiff/.jpg/.jpeg/.bmp`。按实际解码后的通道判断，而非按扩展名判断透明度。

- 默认 `--allow-rgb`：RGB 自动补 alpha=1，与 RGBA 一起进入主重建损失。
- `--no-allow-rgb`：严格要求文件含 alpha/透明信息；不接受无 alpha 的 RGB。带 alpha 但完全不透明的文件仍有效。
- RGBA 保留 alpha。缩放在预乘 alpha 空间完成，然后还原 straight RGBA；半透明图片等比例缩放后透明补边。
- 完全不透明图片等比例缩放后用不透明中灰色补边，不会因补边产生透明区域。
- RGB 和 alpha 最终均归一化为 `[-1,1]`。这是 8-bit 训练路径，不保留 AVIF/TIFF 原始高位深或 HDR；多帧文件仅取首帧。

混合采样按文件数量比例进行，不自动平衡 RGB/RGBA。默认不开水平翻转，适合文字和图标；可用 `--flip` 开启。
省略 `--val-dir` 会固定种子划分 5% 验证集。实际文件列表写入 `split.json`，训练和验证文件不能重叠。

## 启动训练

沿用已有 Accelerate 配置：

```bash
accelerate launch train_rgba_vae.py train \
  --train-dir /data/mixed/train \
  --val-dir /data/mixed/val \
  --output /checkpoints/krea2-rgba \
  --resolution 256 --batch-size 1 --grad-accum 4 \
  --compatibility --allow-rgb --checkpointing
```

精度默认继承 Accelerate，推荐 BF16；可显式 `--precision bf16`。有效 batch = GPU 数 × 每卡 batch × 累积步数。
默认 32,000 次成功 optimizer updates，AdamW，学习率 `1.5e-5`，5% warmup 后 cosine，梯度裁剪 1.0。每 500 步验证，每 1,000 步保存。
256 分辨率、batch 1 是启动设置，未测量真实 Krea 显存；兼容性训练增加前向、反向及教师占用。`--checkpointing` 可减少部分激活占用，不代表固定的显存保证。

## 功能开关

| 功能 | 默认 | 开启/设置 | 关闭 |
|---|---|---|---|
| 兼容性总开关 | 开 | `--compatibility` | `--no-compatibility`，不加载教师 |
| 不透明输入的编码分布对齐 | `0.001` | `--ref-kl-weight 0.001` | `--ref-kl-weight 0` |
| 旧 latent 的 RGB 解码保持 | `1.0` | `--rgb-distill-weight 1` | `--rgb-distill-weight 0` |
| 旧 RGB latent 输出 alpha≈1 | `0.1` | `--opaque-alpha-weight 0.1` | `--opaque-alpha-weight 0` |
| 额外 RGB 语料回放 | 未指定目录则不用 | `--rgb-replay --rgb-replay-dir /data/rgb/train` | `--no-rgb-replay` 或不指定目录 |
| 兼容性验证 | 开，受总开关控制 | `--compat-validation` | `--no-compat-validation` |
| 主训练集接受 RGB | 开 | `--allow-rgb` | `--no-allow-rgb` |
| 主重建的额外 alpha L1 | `0` | `--alpha-weight 0.1` | `--alpha-weight 0` |
| LPIPS | `0.5` | `--lpips-weight 0.5` | `--lpips-weight 0` |
| 激活重计算 | 关 | `--checkpointing` | 不传该参数 |

单项权重为 0 时跳过对应损失；无需编码对齐时跳过学生的额外 encoder，无需 RGB/alpha 保持时跳过学生的额外 decoder，无需 RGB 蒸馏时跳过教师 decoder。主 RGBA 编解码始终执行。
如果所有兼容性权重都为 0，但兼容性验证仍开启，教师只为验证加载。

仅做原来的 RGBA 重建训练：

```bash
accelerate launch train_rgba_vae.py train \
  --train-dir /data/mixed/train --val-dir /data/mixed/val \
  --output /checkpoints/rgba-only --no-compatibility
```

只保留旧 latent 的 RGB 解码约束，关闭编码对齐和 alpha≈1：

```bash
accelerate launch train_rgba_vae.py train \
  --train-dir /data/mixed/train --output /checkpoints/rgb-preserve \
  --compatibility --ref-kl-weight 0 --rgb-distill-weight 1 --opaque-alpha-weight 0
```

## 教师和 RGB 回放

`--reference-model` 默认取 `--model`，必须为原始三通道 RGB VAE。首次从官方模型训练时无需额外指定。
若从已训练的 RGBA 权重继续微调，明确指定原始教师：

```bash
accelerate launch train_rgba_vae.py train \
  --model /checkpoints/old/final/vae \
  --reference-model /models/krea2/vae \
  --train-dir /data/mixed/train --output /checkpoints/next-run
```

使用 Diffusers VAE 目录（`config.json` + 权重），不支持直接加载 ComfyUI 单文件。
可用 `--reference-subfolder`、`--reference-revision` 指定教师来源。教师冻结，不进入 optimizer 或 ZeRO 分片；每卡持有一份，随批次生成目标。续训记录教师权重指纹。

未提供额外 RGB 目录时，每个主样本生成一张黑/白背景合成的 RGB，作为兼容性训练输入。透明前景本身不被强行对齐到两个不同背景。
提供 `--rgb-replay-dir` 时，每个兼容性样本有 50% 概率来自额外 RGB 语料，其余仍为当前主样本的黑/白合成图；额外图像做正方形中心裁剪。额外语料只用于兼容性分支，主目录中的 RGB 则直接参与主重建。
可用 `--rgb-replay-val-dir` 提供额外验证集，省略则从回放目录划出 5%。不能将主验证图片放入回放训练集。

## 网络和损失

三转四通道时，完整复制预训练参数：encoder 新增 alpha 输入权重为 0；decoder 新增 alpha 输出权重为 0、bias 为 1。保持原 causal Conv3d 和 padding。
latent 通道数、空间压缩、`latents_mean` / `latents_std` 均保持原配置，不为 RGBA 重新估计。训练前检查学生/教师接口一致。

默认总损失：

```text
L = ABMSE + 0.5 * LPIPS_black_white + 1e-6 * KL_standard
  + 0.001 * KL_reference + 1.0 * RGB_distillation + 0.1 * Opaque_alpha_MSE
```

主重建项：ABMSE 是背景 `b~Uniform([0,1]^3)` 下精确期望合成 MSE；LPIPS 在黑/白背景合成图上求平均。标准 KL、采样指数、全部损失均用 FP32。

兼容性项：

1. `KL_reference = KL(q_new(z | [rgb, alpha=1]) || q_old(z | rgb))`，对角高斯逐元素取均值。此方向沿用 [AlphaVAE reference KL 的设计](https://arxiv.org/html/2507.09308v1#S4.SS2.SSS3)。黑/白背景每次随机采一个，作为期望的采样估计。
2. 从冻结教师 posterior 采样 **同一个旧 latent** `z_old`，比较 `D_new(z_old)[:3]` 与 `D_old(z_old)`。不能用学生自己的 latent 替代，否则 encoder/decoder 可以一起漂移。
3. 对 `D_new(z_old)` 的 alpha 施加 alpha=1 的 MSE。此约束只作用于旧 RGB latent，不作用于透明主样本的重建目标。

RGB 解码蒸馏、alpha 保持和额外 RGB 回放是针对本需求加入的工程设计；上面的权重是未经过完整 Krea 实验调优的起点。不是 AlphaVAE 的逐项复现，未实现 GAN；ABMSE 使用均匀背景矩与 mean reduction，不能照搬论文 sum reduction 的权重。

QwenImage 公共 `decode()` 会 clamp 到 `[-1,1]`。训练使用同一 decoder 的未裁剪单帧路径，避免输出越界时梯度被硬裁剪；教师蒸馏目标也取未裁剪值。验证/推理仍裁剪。
仅支持单帧 `[B,4,1,H,W]`。重计算使用非重入 checkpoint、缓存重置和 RNG 保留，不调用上游未实现的逐块 checkpointing。

## Accelerate / ZeRO-2

已有 DeepSpeed 配置应满足：

- `zero_optimization.stage: 2`，目前不支持 ZeRO-3 或 optimizer offload。
- JSON 不设置 `optimizer/scheduler`，由脚本提供 AdamW + cosine。
- microbatch、累积步数、总 batch、梯度裁剪用 `auto` 或与脚本参数一致。
- 单帧跳过 temporal-only 卷积，脚本设置 `ignore_unused_parameters: true`。

学生的所有可训练分支都在 prepared model 的 `forward()` 内，使用 `accelerator.accumulate/backward`。不额外除以累积步数，scheduler 每次成功更新只前进一次。
每轮 sampler 重复补齐随机前缀，保证所有 rank 都有完整累积组。全部 rank 参与 checkpoint 保存/恢复，日志和导出文件仅由主进程写入；输出目录需各 rank 可见。

使用附带配置，在脚本目录执行：

```bash
accelerate launch --config_file accelerate_zero2.yaml --num_processes 4 \
  train_rgba_vae.py train \
  --train-dir /data/mixed/train --val-dir /data/mixed/val \
  --output /checkpoints/krea2-rgba --batch-size 1 --grad-accum 4
```

或 Bash 启动器（数据/输出使用绝对路径）：

```bash
TRAIN_DIR=/data/mixed/train VAL_DIR=/data/mixed/val \
OUTPUT_DIR=/checkpoints/krea2-rgba NUM_GPUS=4 GRAD_ACCUM=4 \
bash launch_train.sh --compatibility --checkpointing
```

## 验证、保存与续训

输出包括 `training_args.json`、`split.json`、`metrics.jsonl`、`checkpoint-N/{vae,state,progress.json,training_args.json}`、`best/vae`、`final/vae`、`validation/step-N/`。

主验证报告九种背景的合成 MSE/聚合 PSNR 和 alpha MAE。兼容性验证报告：

- `compat_rgb_psnr / compat_rgb_mse`：同一个教师 posterior mean，经过新旧 decoder 后 RGB 的差异。
- `compat_alpha_mae`：旧 latent 解码 alpha 相对 1 的偏差。
- `compat_ref_kl`：不透明输入编码分布的差异。

预览同时保存旧 latent 的教师 RGB、学生 RGB、学生 RGBA。**这些是编码器 latent 上的兼容性代理指标，尚未覆盖 DiT 生成 latent 的全部分布。** `best/vae` 仍按主验证合成 MSE 选取，不代表该 checkpoint 的旧 DiT 兼容性最好；应结合兼容性指标及真实生成对照选模型。

保持原训练参数，再附加 `--resume /checkpoints/run/checkpoint-0001000` 可恢复 optimizer master/ZeRO 分片、scheduler、RNG 和 epoch/batch 游标。world size、数据、教师指纹、训练开关/权重须一致；max-steps 也须一致。单独导出的 VAE 可能是 BF16/FP16。
如要切换功能、改变配方或从上一版脚本的 checkpoint 迁移，用其 `vae/` 作为 `--model`，新建输出目录；不要恢复不匹配的旧训练状态。重跑旧 step 会追加日志并覆盖同名导出。

## 独立 RGB/RGBA 编解码

```bash
python train_rgba_vae.py encode \
  --model /checkpoints/krea2-rgba/final/vae \
  --input /data/example.avif --output /data/example.latent.safetensors --precision bf16

python train_rgba_vae.py decode \
  --model /checkpoints/krea2-rgba/final/vae \
  --input /data/example.latent.safetensors --output /data/decoded.png --precision bf16

python train_rgba_vae.py reconstruct \
  --model /checkpoints/krea2-rgba/final/vae \
  --input /data/photo.avif --output /data/rgb.png --precision bf16 --rgb-output
```

`--rgb-output` 只取 decoder 的 RGB 三通道，不做 alpha 合成；省略时保存 RGBA。RGB 输入默认补 alpha=1。
编码保存 raw VAE latent，不做 DiT 归一化；自动补到 8 的倍数并记录原尺寸。decode 必须使用与 encode 相同的训练权重。大图推理可用 `--tiling`，可能改变 tile 边缘结果。

## 在原 Krea 2 DiT 上验证

附带 `krea2_latent_bridge.py`，显式处理官方 Diffusers `Krea2Pipeline` 的打包顺序和归一化。该管线 `output_type="latent"` 返回 packed/normalized latent，不能直接传入 VAE `decode()`。
原定义为 `z_normalized = (z_raw - mean) / std`；解码须还原 `z_raw = z_normalized * std + mean`，保持原 mean/std。[官方管线源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/krea2/pipeline_krea2.py)

在已有、未经训练的 `pipe` 上执行（无需替换或训练它的 DiT）：

```python
import torch
from diffusers import AutoencoderKLQwenImage
from train_rgba_vae import save_image
from krea2_latent_bridge import decode_krea_latents

# pipe 是你已加载的官方 Diffusers Krea2Pipeline。
# 保持原来的生成参数。height/width 使用实际生成尺寸，须为 16 的倍数。
height, width = 1024, 1024
packed = pipe(prompt="a red ceramic cup on a table", height=height, width=width,
              generator=torch.Generator("cuda").manual_seed(42),
              output_type="latent").images

new_vae = AutoencoderKLQwenImage.from_pretrained(
    "/checkpoints/krea2-rgba/final/vae", torch_dtype=torch.bfloat16
).to("cuda").eval()
rgb = decode_krea_latents(new_vae, packed, height, width, rgb_only=True)
save_image(rgb[0] * 2 - 1, "/data/old-dit-new-vae-rgb.png")
```

用同一份 `packed`、原 RGB VAE、相同计算精度再解码作对照，避免不同种子干扰比较。保存/离线搬运这些 latent 时同时记录尺寸；这是实际评估所需输入，本训练脚本不会自动运行 DiT。
桥接工具只针对上述 Diffusers 格式，不接受 ComfyUI latent 对象。若原管线有 CPU offload，单独对照解码前需自行确保原 VAE 在合适设备上。

后续适配 DiT 时冻结此 RGBA VAE，使用新 encoder 编码 RGBA，再调用 `pack_krea_latents(raw, new_vae)` 获取同一接口的训练 target；不要增加 DiT 输入/输出通道或重新计算 latent mean/std。本项目不包含 DiT/LoRA 训练器。

## 验证记录

本地验证中，10 项 CPU 测试通过：包括真实 AVIF RGB/RGBA 文件读取、RGB 补边 alpha=1、reference KL 公式与教师冻结、兼容性分支开关、总开关禁用教师、缓存/重计算梯度一致、短训练与断点恢复逐值一致，以及 packed/raw latent 接口和独立编解码。
其余测试覆盖 ABMSE、RGB 通道转换保真、alpha 梯度、LPIPS 连接、全局累积采样。

环境：PyTorch 2.14.0、Diffusers 0.39.0、Accelerate 1.14.0。使用缩小通道数的真实 QwenImage 架构和合成图片，LPIPS 使用随机 AlexNet 以避免下载，仅验证连接。
**未运行完整 Krea 权重的 Linux CUDA/ZeRO-2 训练，未验证实际旧 DiT 画质保持和后续少样本 RGBA 生成效果。** 没有附带训练完成的 RGBA 权重。
