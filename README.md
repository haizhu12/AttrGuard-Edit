# AttrGuard-Edit

AttrGuard-Edit is a runtime-only research snapshot of bounded, rejectable
attribute-preservation feedback for reference-image editing. It extends the
Boogu-Image Edit runtime while retaining the internal `boogu` package and class
names required by compatible checkpoints.

This repository intentionally contains **no model weights, datasets, sample
images, generated outputs, experimental records, or training artifacts**.

## Method status

When an `AttributeControlRequest` is supplied, the paper-aligned defaults are:

- local proxy feedback: `feedback="local"`;
- RGB geometry observation: `observer_profile="geometry"`;
- finite-precision proposals: `proposal_mode="quantized"`.

Attribute control remains opt-in. The standard CLI does not infer constraints
automatically, and this repository does not include a trained or independently
calibrated semantic attribute predictor. The optional bridge in
`utils/predict_attribute_constraints.py` requires a separately supplied
predictor checkpoint and calibration metadata.

The final image path is VAE decode → bilinear resize → image post-processing.
It does not copy source pixels back through a strict-core/feather-band blend or
select among multiple fully decoded output candidates.

## Installation

Python 3.12 and a CUDA-capable PyTorch environment are recommended. For the
standard non-FP8 runtime on Windows or Linux, install the CUDA 12.8 PyTorch trio
from the PyTorch index first, then install the remaining dependencies from
PyPI:

```powershell
python -m pip install -r requirements/torch2.11-cu128.txt
python -m pip install -e .
```

The CUDA requirements file intentionally contains only `torch`, `torchvision`,
and `torchaudio`. In particular, `torchao` is resolved by the second command
from PyPI so that the PowerShell workflow remains installable on Windows.

The FP8 execution path additionally requires Triton kernels and is supported by
this package only on Linux x86_64 (including a compatible WSL2 CUDA setup). The
Windows example below therefore uses the non-FP8 Edit checkpoint. No
`triton-windows` package has been validated or declared by this snapshot.

Obtain a compatible **Boogu-Image Edit** checkpoint separately and keep it
outside this repository. On supported Linux/WSL2 systems, the optional FP8 path
requires a local checkpoint directory whose name identifies it as an Edit FP8
checkpoint.

Only load checkpoints and custom model code from sources you trust and have
audited. The runtime enables Hugging Face `trust_remote_code=True` for required
model components, so a malicious or substituted checkpoint repository can
execute Python code in the inference process.

If optional DashScope rewriting is enabled, set `DASHSCOPE_API_KEY` in the
environment instead of passing the key on the command line. The upstream
rewriting path prints original and rewritten instructions to standard output
and can optionally save them, so treat console logs and saved prompts as
sensitive data.

## Standard editing CLI

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

The CLI performs the base reference-image edit. Paper-method control is exposed
through the programmatic `attribute_request` argument to `inference.run` and
requires fixed, source-aligned `AttributeConstraints` supplied by the caller.

```python
from boogu.attributes.control import ControlConfig
from boogu.attributes.runtime import AttributeControlRequest

config = ControlConfig()
request = AttributeControlRequest(constraints)  # caller-provided constraints

assert config.feedback == "local"
assert config.proposal_mode == "quantized"
assert request.observer_profile == "geometry"
```

## Verification boundary

Before this snapshot was packaged, the full source workspace passed 1,781
tests on 2026-09-21. Those research tests and their generated fixtures are not
included in this runtime-only repository. The mechanism remains research code;
the snapshot does not claim a trained end-to-end semantic predictor or a
demonstrated general image-quality improvement.

## License and attribution

The combined runtime snapshot is distributed under GPL-3.0-only because it
contains TaylorSeer-derived components. Apache-2.0, BSD-3-Clause, and MIT
components retain their original terms; see [NOTICE](NOTICE) and the texts in
[LICENSES](LICENSES). Model checkpoints have their own licenses and are not
distributed here. The repository-level GPL-3.0 text is in [LICENSE](LICENSE).
