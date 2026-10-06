"""Short launcher for the current CLI; supply image, instruction, adapter and permissions.

Run with --help for all options. No experimental weights or source image are
selected implicitly. Pass --method baseline explicitly for the legacy workflow.
Required source input: --input_image_paths SOURCE.png --instruction "your edit".
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_command(argv):
    """Provide local FP8 defaults, then forward user options unchanged."""
    return [
        sys.executable,
        str(ROOT / "inference.py"),
        "--pretrained_pipeline_name_or_path",
        str(ROOT / "models/Boogu-Image-0.1-Edit-fp8-reuse"),
        "--use_fp8_weights",
        "True",
        "--num_inference_steps",
        "50",
        "--text_guidance_scale",
        "1.0",
        "--seed",
        "42",
        "--device",
        "cuda:0",
        *argv,
    ]


def main() -> None:
    subprocess.run(build_command(sys.argv[1:]), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
