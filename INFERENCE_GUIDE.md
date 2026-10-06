# AttrGuard-Edit 0.2.1 推理指南

安装、底座下载与第一次运行见 [README_CN.md](README_CN.md)。本页说明正式入口、输出检查与旧命令迁移；权限制作与训练数据格式见[输入准备指南](docs/source-conditioned-generation.md)。

## 当前入口

`inference.py` 默认 `--method source-conditioned`。必须提供源图、非空指令、正式 adapter 和源图绑定的权限文件。当前没有自动选择的权重或自动权限预测器。

已准备好 README 中的文件后，PowerShell 命令如下：

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

FP8 的通用安装条件是 Linux x86_64 或兼容 WSL2；上面的反引号是 PowerShell 续行符。原生 Windows 的通用安装应使用标准底座路径 `models/Boogu-Image-0.1-Edit` 与 `--use_fp8_weights False`。标准权重的显存需求更高，当前方法没有跨 GPU 的显存保证。

快捷入口可运行同一任务：

```powershell
python inference_ti2i_simple.py `
  --pretrained_pipeline_name_or_path models/Boogu-Image-0.1-Edit-fp8 `
  --input_image_paths inputs/source.png `
  --instruction "只把外套改为蓝色，保持其他内容不变。" `
  --preservation_adapter checkpoints/current/full_adapter.pt `
  --preservation_permissions inputs/permissions.npz `
  --output_image_path outputs/my_edit_simple_seed42.png
```

包装器默认使用本地研究 `*-fp8-reuse` 目录；这里显式覆盖为官方下载目录。两条命令均使用当前 Full 路径。

## 参数与输出

| 参数 | 当前路径的要求 |
| --- | --- |
| `--input_image_paths` | 恰好一张本地图片 |
| `--instruction` | 非空编辑指令，与权限文件一致 |
| `--preservation_adapter` | 匹配底座配置的正式 `.pt` |
| `--preservation_permissions` | 绑定源图 RGB 的 `.npz` |
| `--height / --width` | 不填时使用源图尺寸；填写时必须与源图一致，均为 16 的倍数 |
| `--text_guidance_scale` | 默认 `1`，允许有限值 `>=1`；等于 1 时跳过负向 DiT |
| `--num_inference_steps` | 默认 `50`，正整数 |
| `--seed` | 主入口默认 `0`，快捷入口默认 `42`；示例明确使用 `42` |
| `--reference_seed` | 默认为 `seed+1`，在最大合法种子处回绕；必须与主种子不同 |
| `--scheduler` | 仅 `euler` |
| `--device / --dtype` | CUDA / `bf16`；VAE 自动使用 FP32 |
| `--output_image_path` | 未使用的 `.png` 路径，同名 `.png.json` 也必须不存在 |

成功运行写出图片和来源记录。JSON 包含版本、输入及 adapter/权限哈希、种子、画布、采样参数，以及 `negative_dit_branch`；CFG=1 时该字段应为 `skipped`。它记录底座配置/索引，**不包含所有大型底座权重的逐文件哈希**，因此还需自行保留实际模型 revision 和运行环境。

检查源图、编辑结果以及非目标区域，不能只凭进程成功或来源记录宣布保持质量达标。当前研究 adapter 的训练规模为一张真实照片、96 次更新。

## 常见问题

| 报错或现象 | 处理 |
| --- | --- |
| 缺少 preservation 输入 | 提供源图、指令、正式 adapter 和权限；不会回退旧方法 |
| adapter metadata 不匹配 | 使用匹配底座的 Full adapter；仅对受支持的研究格式运行导出工具 |
| `source_sha256 does not match` | 源图已改变或用了其他图片的权限；重新核对源图并制作权限 |
| 权限 shape/type 错误 | 使用浮点 `[objects,5,H,W]`，字符串字段须为 Unicode；详细格式见输入准备指南 |
| `overlapping keep/modify` | 修正同一属性位置上的 K/M 冲突，包括跨对象冲突 |
| `resizing is unsupported` | 预先准备符合尺寸的源图并重新生成权限；不要只改 `height/width` |
| `would resize the source` | 提高对应输入像素/边长上限，使其容纳已准备的画布 |
| `fresh paths` | 更换输出文件名；同时检查图片和 JSON 是否已存在 |
| `does not support explicit` | 删除当前路径不支持的选项，包括显式传入的旧选项 `False` |
| CUDA/BF16 不可用 | 检查 CUDA wheel、驱动与设备能力 |
| CUDA OOM | 释放其他 GPU 任务；需要较小画布时同步重做源图与权限。不要给当前路径添加旧 offload 开关 |
| FP8/Triton 加载失败 | 核对底座格式与平台依赖，或改用标准 Edit 权重及 `--use_fp8_weights False` |

## 旧命令迁移

旧 Edit 运行方式保留为显式 baseline。下面是旧方法的独立调用，不加载 Full adapter：

```powershell
$env:device = "cuda:0"
python inference.py `
  --method baseline `
  --pretrained_pipeline_name_or_path models/Boogu-Image-0.1-Edit-fp8 `
  --use_fp8_weights True --dtype bf16 --device cuda:0 `
  --input_image_paths inputs/source.png `
  --instruction "只把外套改为蓝色，保持其他内容不变。" `
  --num_inference_steps 50 --text_guidance_scale 1 `
  --seed 42 --reference_seed 43 `
  --output_image_path outputs/baseline_seed42.png
```

重放历史实验时使用原协议指定的 CFG、步数、代码与权重；上例是通用旧方法调用，不是历史结果的逐像素复现命令。FAR、旧属性修正、缓存、offload、改写和批量属于旧路径能力，不能混入当前 source-conditioned 命令。

`inference_turbo.py` 只用于匹配的 Edit-Turbo 工作流，不支持当前 adapter。普通 Edit 权重不能当成四步蒸馏权重。

如果已有研究底座使用 Windows 目录联接，保留其共享目标。特别是 `models/Boogu-Image-0.1-Edit-fp8-reuse` 依赖的 `models/Boogu-Image-0.1-Turbo-fp8` 不可删除或移动。
