# AttrGuard-Edit 0.2.1：输入准备与训练

本页提供正式入口可读取的文件格式与操作步骤。安装和推理见 [README_CN.md](../README_CN.md)，错误处理见 [INFERENCE_GUIDE.md](../INFERENCE_GUIDE.md)。

当前 Full 架构与已选研究权重保持一致；CFG=1 时跳过负向 DiT。版本集成本身没有重新训练 adapter。现有本地研究检查点仅在一张真实训练照片上更新 96 次，尚无公开的通用 adapter 或自动属性权限预测器。

<a id="permissions"></a>

## 1. 制作源图绑定的权限文件

源图先进行 EXIF 方向修正并转 RGB；宽高须为 16 的倍数。所有标注必须对齐这一画布。权限来自源图与编辑指令，不能从配对目标图反推。

`permissions.npz` 通过 `allow_pickle=False` 读取，仅允许下列字段：

| 字段 | 格式 |
| --- | --- |
| `schema` | Unicode 字符串标量 `boogu_preservation_permissions_v1` |
| `attributes` | Unicode 字符串数组，固定为 `["color","structure","texture","glyph","pixel"]` |
| `source_sha256` | Unicode 字符串标量；对 EXIF 修正后 RGB 图像的 `tobytes()` 求 SHA256 |
| `keep` | 浮点 `[objects,5,H,W]`，有限值且在 `[0,1]` |
| `modify` | 与 `keep` 同形状的浮点数组，有限值且在 `[0,1]` |
| `reference_valid` | 可选，同形状布尔数组；省略时参考有效 |

对象数至少为 1。同一属性和位置上的 keep/modify 不能重叠，包括跨对象重叠。两者都为零表示未知；缺失标签不能自动补成 keep。参考无效处的 keep 也视为未知。

### 用逐属性标注 PNG 制作文件

为 `inputs/source.png` 标注同画布的灰度图，保存到 `inputs/permissions/`。文件名是 `color_keep.png`、`color_modify.png`，以及 structure、texture、glyph、pixel 的同名组合。白色表示权重 1，黑色表示 0，灰色表示中间权重；未提供的图保持全零。

每个属性需独立审核。例如“只改外套颜色”不能直接把外套 mask 复制成结构、纹理、字形和像素的修改权限。某个属性没有可靠标注时，将其保留为未知。

把以下内容保存为 `make_permissions.py`，在仓库根目录运行 `python make_permissions.py`：

```python
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from boogu.preservation.runtime import load_permissions, source_rgb_sha256

source_path = Path("inputs/source.png")
destination = Path("inputs/permissions.npz")
attributes = ("color", "structure", "texture", "glyph", "pixel")
with Image.open(source_path) as opened:
    source = ImageOps.exif_transpose(opened).convert("RGB")
if source.width % 16 or source.height % 16:
    raise ValueError("Prepare a source canvas divisible by 16 first.")

maps = {}
loaded = 0
for role in ("keep", "modify"):
    values = np.zeros((1, 5, source.height, source.width), dtype=np.float32)
    for channel, attribute in enumerate(attributes):
        path = Path("inputs/permissions") / f"{attribute}_{role}.png"
        if path.exists():
            with Image.open(path) as opened:
                mask = opened.convert("L")
                if mask.size != source.size:
                    raise ValueError(f"Mask canvas differs from source: {path}")
                values[0, channel] = np.asarray(mask, dtype=np.float32) / 255
            loaded += 1
    maps[role] = values
if not loaded:
    raise ValueError("Provide reviewed per-attribute masks before continuing.")

destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("xb") as stream:
    np.savez_compressed(
        stream,
        schema=np.array("boogu_preservation_permissions_v1"),
        attributes=np.array(attributes),
        source_sha256=np.array(source_rgb_sha256(source)),
        **maps,
    )
load_permissions(destination, source)
print(f"Validated {destination}; loaded {loaded} attribute masks.")
```

该脚本只打包已有人工标注，不预测语义权限。输出文件必须未使用；若最后校验报告冲突，修正标注后另选输出名重做。多对象数据可自行构造对象维度；不要改变属性顺序。

已有权限文件可单独检查：

```bash
python -c "from PIL import Image; from boogu.preservation.runtime import load_permissions; load_permissions('inputs/permissions.npz', Image.open('inputs/source.png')); print('permissions OK')"
```

## 2. 选定或导出 adapter

正式推理接受匹配底座配置的 `{metadata, adapters}` adapter 文件。较早的 Full 研究封装需显式迁移：

```bash
python export_preservation_adapter.py --source path/to/full_research_adapter.pt --base-config models/Boogu-Image-0.1-Edit-fp8/transformer/config.json --output checkpoints/current/full_adapter.pt
```

源文件必须是受支持的 Full 研究检查点。工具验证结构与参数、拒绝其他消融版本、保持所有 adapter 参数数值不变，并写出 `full_adapter.pt.json`。输出和 JSON 均须为新文件。这是格式导出，不增加训练更新，也不构成新的质量验证。

<a id="training"></a>

## 3. 准备真实配对训练数据

`train_preservation.py` 提供 validate、prepare、train 三个子命令。它训练新的 Full adapter；现有研究权重的版本集成不需要重新运行训练。

每条配对需包含同画布、逐像素对齐的源图和目标图、明确的编辑指令、第 1 节的源图权限，以及独立的训练监督文件。目标图只用于监督，不用于制作推理权限。

下面是 `data/paired_train.json` 的完整格式示例。示例中的照片、标注和权限需自行准备；文件路径相对于 manifest 所在目录：

```json
{
  "schema": "boogu_preservation_pairs_v1",
  "pairs": [
    {
      "pair_id": "coat_001_blue",
      "source": "coat_001/source.png",
      "target": "coat_001/target.png",
      "instruction": "Change only the jacket to blue; preserve everything else.",
      "permissions": "coat_001/permissions.npz",
      "supervision": "coat_001/supervision.npz",
      "split": "train",
      "source_group": "photo_coat_001",
      "admitted": true,
      "same_frame": true
    }
  ]
}
```

字段必须完整，不能额外添加字段。`admitted=true` 表示数据已审核；`same_frame=true` 是调用方对图像配准的明确确认，程序的尺寸检查不能替代人工审核。相同源照片派生的配对使用稳定的 `source_group`，并在整个数据集划分时保留在同一 split。

训练只接受 train-only manifest；validation/test 数据需独立保留。程序检查所提供 manifest 内的重复 ID、重复配对与图像/分组泄漏，不会替你审计仓库外的所有数据。

### 制作训练监督 NPZ

`supervision.npz` 使用下列布尔 `[H,W]` 数组，不能混入推理权限文件：

| 字段 | 用途 |
| --- | --- |
| `allowed_edit` | 已审核的允许编辑区域 |
| `edit_core` | 需要完成编辑的核心区域，必须是 allowed_edit 的子集 |
| `protected_roi__名称` | 可选，需要单独评估保持的区域；名称仅使用字母、数字、下划线和短横线 |

将人工审核的二值 mask 保存为 `data/coat_001/allowed_edit.png`、`edit_core.png`，白色为真、黑色为假。以下脚本可保存为 `make_supervision.py` 后运行：

```python
from pathlib import Path

import numpy as np
from PIL import Image

folder = Path("data/coat_001")
masks = {}
for name in ("allowed_edit", "edit_core"):
    with Image.open(folder / f"{name}.png") as opened:
        values = np.asarray(opened.convert("L"))
    if not np.isin(values, (0, 255)).all():
        raise ValueError(f"{name} must contain only black and white pixels.")
    masks[name] = values == 255
if masks["allowed_edit"].shape != masks["edit_core"].shape:
    raise ValueError("Supervision masks must have identical canvases.")
if np.any(masks["edit_core"] & ~masks["allowed_edit"]):
    raise ValueError("edit_core must be inside allowed_edit.")
with (folder / "supervision.npz").open("xb") as stream:
    np.savez_compressed(stream, **masks)
print("Saved supervision.npz; run manifest validation next.")
```

需要保护 ROI 时，在保存前加入例如 `masks["protected_roi__face"]`，其值必须是同画布布尔数组。允许空区域，但它们不会提供对应区域的训练信号。

## 4. 校验、缓存、训练

先校验文件与数据约束，不加载 GPU 模型：

```bash
python train_preservation.py validate --manifest data/paired_train.json
```

FP8 底座的准备和训练命令如下。两步均需前述 CUDA/BF16 环境；`--use-fp8-weights` 是训练入口的开关，不要在其后加 `True`。

```bash
python train_preservation.py prepare --manifest data/paired_train.json --base models/Boogu-Image-0.1-Edit-fp8 --cache training_cache/coat_v1 --use-fp8-weights --device cuda:0 --reference-seed 43 --max-sequence-length 1280
python train_preservation.py train --manifest data/paired_train.json --base models/Boogu-Image-0.1-Edit-fp8 --cache training_cache/coat_v1 --output checkpoints/trained/full_adapter.pt --use-fp8-weights --device cuda:0 --reference-seed 43 --max-sequence-length 1280 --steps 96 --lr 0.0003 --seed 42
```

使用标准底座时改为标准 Edit 路径，并在两条命令中删除 `--use-fp8-weights`。缓存目录必须是新目录；prepare 完整成功后才写出 `index.json`。train 使用同一 manifest、底座和缓存设置，并校验绑定文件，不能拿修改过的数据继续使用旧缓存。

成功训练输出正式 `full_adapter.pt` 与 `full_adapter.pt.json`，前者可直接作为推理的 `--preservation_adapter`，无需再导出。输出路径必须未使用。96 步和学习率是运行示例，不是经过广泛验证的最优训练方案。

当前入口只训练 Full；不提供 resume、分布式训练、梯度累积或自动评估。训练成功和损失下降不能替代独立真实照片上的编辑成功与非目标属性保持评估。

## 5. 版本与历史记录

0.2.1 的日常推理与训练不依赖 `outputs/` 中的实验 Python 文件或旧缓存。历史冻结协议、合成小样本验证、NASA 宇航员展示和旧指标保留其原版本含义，不能直接当作当前正式入口的效果证明。当前文档不把这些历史实验列为推荐训练流程。
