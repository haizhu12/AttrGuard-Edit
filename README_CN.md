# AttrGuard-Edit 0.2.2

[English](README.md)

AttrGuard-Edit 是研究用途的图像编辑运行时，目标是在完成指令编辑时保持其他属性。0.2.2 存档当前 Full 源图条件推理、配对训练和默认关闭的推理时保持控制；版本变化见 [CHANGELOG](CHANGELOG.md)。

当前默认 CFG=1，此时跳过负向 DiT 分支；Full 结构不变。本次版本集成没有重新训练已有研究适配器，也不表示通用图像质量已经得到验证。

## 1. 运行前需要什么

需要 Python 3.10–3.12、支持 BF16 的 CUDA GPU、完整的本地 Boogu Edit 底座，以及明确选定的 Full adapter。每次编辑还需提供一张源图、编辑指令和绑定该源图的逐属性权限文件。

仓库没有公开发布训练好的 AttrGuard adapter，也没有可直接使用的自动权限预测器。当前本地研究权重只在 **1 张真实训练照片上更新了 96 次**，不能当作经过充分验证的通用编辑模型。底座、研究数据、私有检查点和本地输出不随代码上传。

## 2. 安装环境

在仓库目录执行；已有可用环境可跳过安装。

```powershell
git clone https://github.com/haizhu12/AttrGuard-Edit.git
cd AttrGuard-Edit
conda create -n attrguard-edit python=3.12 -y
conda activate attrguard-edit
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install "huggingface-hub>=0.34,<2"
python -m pip check
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
```

当前方法要求 CUDA、BF16 都输出 `True`。以上是安装步骤，不代表所有平台都已完成全新安装验证。FP8 的通用安装路径使用 Linux x86_64 或兼容的 WSL2；项目只在 Linux x86_64 声明 Triton 依赖。本机已有的 Windows 研究环境不能直接视为原生 Windows 的 FP8 安装教程。

## 3. 准备官方底座

完整权重的官方来源：

| 底座 | 固定 revision |
| --- | --- |
| [Boogu-Image-0.1-Edit](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit) | `0cce2581de53bfdb5d67aa438e39c3d6e7556808` |
| [Boogu-Image-0.1-Edit-fp8](https://huggingface.co/Boogu/Boogu-Image-0.1-Edit-fp8) | `98dad8c1c6ed02e5293f64433e7900fc6f3a4060` |

下面推理命令使用 FP8 版：

```powershell
hf download Boogu/Boogu-Image-0.1-Edit-fp8 --revision 98dad8c1c6ed02e5293f64433e7900fc6f3a4060 --local-dir models/Boogu-Image-0.1-Edit-fp8
```

完整目录包含 `model_index.json`、`mllm/`、`processor/`、`scheduler/`、`transformer/` 和 `vae/`，无需另外下载 Qwen 或 VAE。目录名称需保留 Edit 标识；当前方法拒绝 Base 和 Turbo 底座。

使用标准权重时，改用下面的下载命令，并把推理命令中的底座路径与 `--use_fp8_weights False` 一起修改：

```powershell
hf download Boogu/Boogu-Image-0.1-Edit --revision 0cce2581de53bfdb5d67aa438e39c3d6e7556808 --local-dir models/Boogu-Image-0.1-Edit
```

## 4. 准备源图、权限与 adapter

将照片准备为 `inputs/source.png`。宽高均须为 16 的倍数；不填 `height/width` 时沿用源图尺寸，不会自动缩放。

按[权限文件步骤](docs/source-conditioned-generation.md#permissions)准备 `inputs/permissions.npz`。颜色、结构、纹理、字形、像素分别标注保持或允许修改；未标注保持未知，不能用一个空间 mask 推断全部属性都可修改。

把兼容的正式 adapter 放到 `checkpoints/current/full_adapter.pt`，或在命令中使用自己的路径。已有 **Full 研究格式检查点**时，先显式导出：

```powershell
python export_preservation_adapter.py --source path/to/full_research_adapter.pt --base-config models/Boogu-Image-0.1-Edit-fp8/transformer/config.json --output checkpoints/current/full_adapter.pt
```

`path/to/` 需要替换为自己的实际文件，仓库没有内置该权重。导出必须使用未占用的路径，同时产生 `.pt.json` 来源记录并验证参数数值未改变；这一步不训练模型，也不接受其他消融版本冒充 Full。

如果从真实配对数据重新训练，请按[训练入口与数据格式](docs/source-conditioned-generation.md#training)操作；新训练入口直接保存正式 adapter。

## 5. 正式推理

以下使用 PowerShell 的续行语法；Linux / WSL2 Bash 命令见 [English README](README.md#4-run-the-current-method)。FP8 仍需满足前述平台条件。

```powershell
$env:device = "cuda:0"
python inference.py `
  --method source-conditioned `
  --pretrained_pipeline_name_or_path models/Boogu-Image-0.1-Edit-fp8 `
  --use_fp8_weights True --dtype bf16 --device cuda:0 `
  --input_image_paths inputs/source.png `
  --instruction "只把外套改为蓝色，保持其他内容不变。" `
  --preservation_adapter checkpoints/current/full_adapter.pt `
  --preservation_permissions inputs/permissions.npz `
  --scheduler euler --num_inference_steps 50 --text_guidance_scale 1 `
  --seed 42 --reference_seed 43 `
  --output_image_path outputs/my_edit_seed42.png
```

编辑指令必须与权限文件一致。成功后生成 PNG 和同名 `.png.json` 运行记录，父目录自动创建。重复运行需改成未使用的输出名。

快捷入口 `inference_ti2i_simple.py` 委托给同一正式 CLI，默认 50 步、CFG=1、seed=42、CUDA，底座默认为研究布局 `models/Boogu-Image-0.1-Edit-fp8-reuse`。使用官方完整下载目录时，需显式覆盖 `--pretrained_pipeline_name_or_path`。它不会自动挑选源图、指令、adapter 或权限。

### 实验性保持开关（默认关闭）

在上述命令中添加 `--preservation_keep_mode project` 或 `--preservation_keep_mode source-memory`。省略时仍为 `off`，不改变原推理路径或已有权重。

- `project`：在采样过程中保持明确要求像素不变的区域，作为空间约束对照。
- `source-memory`：在此基础上，逐层引入保护区域的源图特征；要求 CFG=1、负指令为空。存在可保护区域时，每步多执行一次不加载适配器的源图 DiT 前向。

只有整个图像 token 都是 pixel-K、且所有属性均无 modify 的位置才会锁定；混合区域、未知区域以及只有结构/纹理保持要求的位置不会被整块锁住。该功能不是结果像素回贴，也不能保证编辑对象内部的非目标属性不变。它目前是用于验证的可选机制，不是已确认更优的新默认方案。详细边界与运行记录见[实验说明](docs/executable-keep-routing.md)。

10 月 7 日的 18 次固定对照已完成：两种开启模式降低了保留区误差，但 source-memory 也出现了抑制目标替换、保留待删除球的反效果。因此都不提升为默认方案，也不能据此宣称通用属性保留问题已解决。

### 实验性潜变量优化（默认关闭）

在正式源图条件命令中追加：

```text
--preservation_optimization latent --preservation_edit_anchor attribute --preservation_keep_mode off --text_guidance_scale 1
```

仅通过冻结的 FP32 VAE 优化当前图像潜变量，不更新模型权重。省略开关仍关闭优化；`--preservation_edit_anchor rgb` 选择早期 RGB 锚点，`attribute` 使用显式属性权限，只有 pixel-M 时精确回退旧 RGB 规则。属性锚点必须配合 latent 优化；不能与保持路由、CFG≠1 或旧 baseline 混用。

10 月 9 日诊断在一个小狗换色案例中改善了部分保持代理，但主要毛发／面部差异仍在；另五组兼容性对照未改变，不能据此宣称通用结构或身份保真。本轮使用另一份已存在的 MagicBrush pilot（9 对训练、96 次更新），不是前述一张照片的旧权重；两者都不随源码分发。使用与限制见[潜变量优化说明](docs/inference-latent-optimization.md)。本次源码存档不上传新实验图、数据或私有实验执行器。

## 6. 支持范围与旧版迁移

当前路径限定单图、单参考、同画布、Euler、BF16 计算和 FP32 VAE。批量、多参考、变画布、FAR、旧属性修正、缓存加速、hook offload 与指令改写均不支持；参数不兼容时会在生成前报错。

旧 Edit 工作流需显式加入 `--method baseline`，并删除 preservation 参数。`inference_turbo.py` 仍是独立的旧 Edit-Turbo 入口，不能加载当前 adapter。早期 NASA 宇航员或合成展示只记录历史方法，不能证明 0.2.2 的质量；本次不上传新的输出图。

若本机底座通过目录联接复用组件，保留其目标目录。研究环境中的 `models/Boogu-Image-0.1-Edit-fp8-reuse` 依赖 `models/Boogu-Image-0.1-Turbo-fp8`，不能删除后者。

详细参数与排错见 [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)。组合发行版按 GPL-3.0-only 分发，各组件保留原许可；参见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [LICENSES](LICENSES)。模型权重由其所有者另行分发。
