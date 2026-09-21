# AttrGuard-Edit

AttrGuard-Edit 是一个仅包含运行时关键源码的研究快照，用于在参考图像编辑采样过程中施加有界、可拒绝的属性保持反馈。为了兼容既有 Boogu-Image Edit 权重，内部 Python 包名 `boogu`、模型类名和序列化路径保持不变。

本仓库明确不包含模型权重、数据集、样图、生成结果、实验记录、训练缓存或中间脚本。

## 当前方法边界

显式传入 `AttributeControlRequest` 时，论文对齐默认路径为：

- `feedback="local"`；
- `observer_profile="geometry"`；
- `proposal_mode="quantized"`。

属性控制仍为可选功能。普通 `inference.py` 命令不会自动预测属性约束；仓库也没有附带经过独立校准的语义属性预测器权重。可选桥接代码 `utils/predict_attribute_constraints.py` 只有在调用方另行提供训练完成的预测器检查点和校准信息时才能使用。

最终输出路径为 VAE 解码、双线性恢复目标尺寸和图像后处理，不执行“严格核心 + 羽化带 + 源图像素”的末端融合，也不在多张完整解码候选图之间选图。

## 安装

建议使用 Python 3.12 和 CUDA 环境。对于 Windows 或 Linux 上的标准非 FP8 运行路径，先从 PyTorch 专用索引安装 CUDA 12.8 的 PyTorch 三件套，再从 PyPI 安装其余项目依赖：

```powershell
python -m pip install -r requirements/torch2.11-cu128.txt
python -m pip install -e .
```

CUDA requirements 文件有意只包含 `torch`、`torchvision` 和 `torchaudio`；`torchao` 由第二条命令从 PyPI 解析，避免 PowerShell/Windows 安装流程因 PyTorch CUDA 专用索引缺少对应轮子而失败。

FP8 路径还依赖 Triton 内核，本快照仅在 Linux x86_64（包括兼容的 WSL2 CUDA 环境）声明支持。所以下面的 Windows/PowerShell 示例使用非 FP8 Edit 权重；本项目没有声明或验证 `triton-windows`。

模型权重需要单独获取并存放在仓库外。在受支持的 Linux/WSL2 环境中，FP8 推理应传入本地 Boogu-Image Edit FP8 检查点目录。

只加载来源可信且已经审计的检查点和自定义模型代码。运行时会为所需模型组件启用 Hugging Face 的 `trust_remote_code=True`，因此恶意或被替换的检查点仓库可能在推理进程中执行 Python 代码。

如启用可选的 DashScope 指令改写，请通过环境变量 `DASHSCOPE_API_KEY` 提供密钥，不要把密钥直接写入命令行。上游改写路径会将原始和改写后的指令打印到标准输出，并可选择保存这些内容，因此控制台日志与保存的提示词都应按敏感数据处理。

## 基础图像编辑

```powershell
python inference.py `
  --pretrained_pipeline_name_or_path "../external-models/Boogu-Image-0.1-Edit" `
  --use_fp8_weights False `
  --input_image_paths "../input/source.png" `
  --instruction "将目标对象改为棕色，并保持其余内容不变。" `
  --num_inference_steps 25 `
  --text_guidance_scale 4.0 `
  --image_guidance_scale 1.0 `
  --seed 42 `
  --output_image_path "result.png" `
  --device cuda:0
```

这条命令执行基础参考图像编辑。论文方法通过 `inference.run(..., attribute_request=request)` 以编程接口接入，调用方必须提供与源图对齐且来源明确的 `AttributeConstraints`。

发布打包前，完整研究工作区于 2026-09-21 通过 1,781 项测试。测试、夹具和实验数据未收入这个精简仓库。该快照不宣称已经具备可直接使用的端到端语义预测能力，也不据此宣称整体图像质量显著提升。

由于包含源自 TaylorSeer 的组件，本组合运行时快照整体按 GPL-3.0-only 分发。原有 Apache-2.0、BSD-3-Clause 和 MIT 组件继续适用各自条款，具体路径映射与完整许可文本见 [NOTICE](NOTICE)、[LICENSE](LICENSE) 和 [LICENSES](LICENSES)。模型权重不在本仓库分发，并适用其各自许可证。
