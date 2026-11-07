# CPU regression tests

The `CPU tests` GitHub Actions workflow runs on every push and pull request,
and can also be started manually. It uses Ubuntu 24.04, Python 3.12,
CPU-only PyTorch 2.7.1 and torchvision 0.22.1, read-only repository permissions,
and a 20-minute timeout. Checkout does not persist credentials.

## Scope

All ten current `tests/test_*.py` files run, with no exclusions or GPU skips:

| Test file | Coverage |
| --- | --- |
| `test_keep_routing.py` | Latent projection, source-memory block routing, small native transformer smoke test |
| `test_inference_optimization.py` | Frozen-decoder latent optimization, acceptance/backtracking, gradients and no-op behavior |
| `test_edit_anchoring.py` | RGB and attribute anchors, masking, validation and gradients |
| `test_preservation_cli.py` | CLI defaults, option validation, output handling |
| `test_far_cli.py` | FAR CLI parsing, configuration and preflight contracts |
| `test_preservation_checkpoints.py` | Checkpoint export and validation |
| `test_preservation_runtime.py` | Permissions, preflight, scheduler and staged runtime orchestration |
| `test_preservation_training.py` | Synthetic pair manifests, cache preparation and small-model training contracts |
| `test_source_preservation.py` | Conditioning, adapter lifecycle, checkpoint gradients and small native model integration |
| `test_source_preservation_objectives.py` | Continuous preservation loss, support masks and gradients |

Fixtures create their own images, tensors, permission files and checkpoints in
temporary directories. Native transformer tests construct small models locally;
orchestration tests replace pretrained loaders and CUDA checks with test doubles.
CUDA-related validation tests mock CUDA APIs rather than allocate GPU tensors.
No model weights, private datasets, credentials or GPU are required.

## Reproduce locally

From the repository root, in a clean Python 3.12 virtual environment:

```bash
python -m pip install torch==2.7.1+cpu torchvision==0.22.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/cpu-tests.txt
python -m pip check
CUDA_VISIBLE_DEVICES='' device=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --disable-socket tests
```

Dependency installation needs network access. Test execution disables sockets
with `pytest-socket` and enables Hugging Face offline mode, so unexpected download
attempts fail. `python -m pytest` adds the repository root to Python's import path;
the tests can import `boogu` directly and read the root CLI scripts via `Path`/AST.
Installing the full project is unnecessary and would pull production-only GPU
dependencies such as Triton, torchao and kernels. The CPU requirements pin the
direct test dependencies, not every transitive dependency.

## Validation boundary

Passing this suite verifies CPU contracts and small-model behavior. It does not
validate full pretrained Boogu/Qwen/VAE inference, image-edit quality, real-data
training, CUDA BF16/FP8 execution, FlashAttention/Triton kernels, device offload,
GPU memory usage or throughput. Those still need a separate GPU validation run
with the intended model weights and data. Production dependencies and model/CLI
defaults are unchanged by this workflow.
