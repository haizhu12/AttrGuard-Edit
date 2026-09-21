"""Single-object semantic bridge; requires an independently trained checkpoint.

The template parser deliberately rejects unsupported/compound edits. It does not
invent semantic roles from generated targets. No trained artifact is bundled.
"""

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from boogu.attributes.constraints import AttributeConstraints, predicted_weights
from boogu.attributes.control import ControlConfig
from boogu.attributes.runtime import AttributeControlRequest
from boogu.attributes.training import load_attribute_checkpoint


def combine_attribute_diagnostics(request, preparation):
    """Combine static semantic preparation and sampling records after inference.

    The sampler itself consumes fixed constraints, whether manual or precomputed.
    Rename its legacy local flag so it cannot override the end-to-end evidence.
    """
    if (
        not isinstance(preparation, dict)
        or preparation.get("semantic_predictor_precomputed") is not True
    ):
        raise ValueError("a semantic preparation record from the bridge is required")
    control = dict(request.diagnostics)
    executed = control.pop("semantic_predictor_used", False)
    control["semantic_predictor_executed_in_pipeline"] = executed
    return {
        "semantic_predictor_used": True,
        "semantic_preparation": dict(preparation),
        "control": control,
    }


@dataclass(frozen=True)
class OperationRecord:
    operation: str
    attribute: int
    object_phrase: str
    target: str
    protection_clause: str

    def condition_text(self, instruction):
        return (
            instruction + "\n[属性约束分支对象记录]\n"
            f"对象：{self.object_phrase}\n操作：{self.operation}\n"
            f"目标：{self.target}\n显式保护：{self.protection_clause}"
        )


def parse_operation(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("one nonempty Chinese template instruction is required")
    if len(re.findall("改为|替换为|移除", instruction)) != 1:
        raise ValueError("only one unambiguous template operation is supported")
    sentences = [part.strip() for part in instruction.split("。") if part.strip()]
    for clause in sentences[1:]:
        if not re.fullmatch(r"保持.+不变", clause) or re.search(
            r"改为|替换|移除|添加|放大|缩小|旋转|移动|翻转|删除|把|将", clause
        ):
            raise ValueError("additional sentences must be pure keep-unchanged clauses")
    protection = "。".join(sentences[1:])
    head = sentences[0]
    remove = re.fullmatch(r"(?:仅)?移除([^，,]+)(?:[，,]并用.+)?", head)
    if remove and any(word in remove[1] for word in ("字", "标题", "文字")):
        return OperationRecord("remove_text", 3, remove[1], "移除并填补", protection)
    change = re.fullmatch(r"(?:仅)?将(.+?)(?:改为|替换为)([^，,。]+)", head)
    if not change:
        raise ValueError(
            "unsupported operation template; use explicit single recolor/text edit"
        )
    obj, target = change[1].strip(), change[2].strip('“”" ')
    if any(word in obj for word in ("数字", "文字", "字样", "标题")):
        return OperationRecord("replace_text", 3, obj, target, protection)
    if re.fullmatch(
        r"(?:深|浅|亮|暗)?(?:红|橙|黄|绿|蓝|紫|粉|棕|黑|白|灰|金|银)色", target
    ):
        return OperationRecord("recolor", 0, obj, target, protection)
    raise ValueError(
        "unsupported target attribute; geometry-changing edits have no correspondence contract"
    )


def prediction_constraints(
    prediction,
    source,
    operation,
    *,
    threshold,
    unknown_max,
    scales,
    tolerances,
    provenance,
    protected_keep=None,
):
    _validate_protected_keep(operation, protected_keep, source.shape[-2:])
    probabilities = prediction.role_probabilities
    if probabilities.shape[0] != 1:
        raise ValueError("this bridge supports one source and one parsed object")
    keep, modify = predicted_weights(
        probabilities,
        prediction.object_support,
        prediction.applicability,
        threshold=threshold,
        unknown_max=unknown_max,
    )
    canvas = source.shape[-2:]
    # Threshold on the calibrated prediction grid, then fixed nearest projection.
    # Interpolating logits first would silently change confidence decisions.
    keep = F.interpolate(keep.float(), canvas, mode="nearest")
    modify = F.interpolate(modify.float(), canvas, mode="nearest")
    allowed = torch.zeros_like(modify)
    allowed[:, operation.attribute] = modify[:, operation.attribute]
    modify = allowed
    edit = (modify[:, operation.attribute : operation.attribute + 1] > 0).float()
    if protected_keep is not None:
        # These objects are supplied on source coordinates, independently of the
        # target object's predicted support. They are not automatic segmentation.
        keep = torch.cat((keep, protected_keep.detach().to(keep).clone()), dim=0)
    reference_valid = torch.ones_like(keep, dtype=torch.bool)
    # Explicit modification wins over RGB identity. Unsupported attributes are U.
    keep[:, 4] *= modify.sum(1) == 0
    keep[:, operation.attribute] *= edit[:, 0] == 0
    reference_valid[:, 4] &= edit[:, 0] == 0
    reference_valid[:, operation.attribute] &= edit[:, 0] == 0
    if operation.operation in ("replace_text", "remove_text"):
        # Old glyph edges/textures cannot serve as identity targets for new
        # letters. Removing text also reveals background absent from the source.
        incompatible = (
            range(5) if operation.operation == "remove_text" else (1, 2, 3, 4)
        )
        for attribute in incompatible:
            keep[:, attribute] *= edit[:, 0] == 0
            reference_valid[:, attribute] &= edit[:, 0] == 0
    if operation.protection_clause and not keep[1:].any():
        raise ValueError(
            "protected_keep has no compatible protection after edit exclusions"
        )
    return AttributeConstraints(
        source,
        keep,
        edit,
        scales,
        tolerances,
        provenance,
        reference_valid=reference_valid,
    )


def _validate_protected_keep(operation, protected_keep, canvas):
    if protected_keep is None:
        if operation.protection_clause:
            raise ValueError(
                "explicit protection clauses require source-aligned protected_keep; "
                "the single-target predictor cannot resolve other protected objects"
            )
        return
    if (
        not isinstance(protected_keep, torch.Tensor)
        or protected_keep.ndim != 4
        or protected_keep.shape[0] < 1
        or protected_keep.shape[1:] != (5, *canvas)
        or not torch.is_floating_point(protected_keep)
        or not torch.isfinite(protected_keep).all()
        or protected_keep.min() < 0
        or protected_keep.max() > 1
        or (operation.protection_clause and not protected_keep.any())
    ):
        raise ValueError(
            "protected_keep requires nonempty explicit [objects,5,H,W] weights"
        )


@torch.no_grad()
def predict_attribute_request(
    pipeline,
    source_image,
    instruction,
    checkpoint,
    *,
    threshold,
    unknown_max,
    scales,
    tolerances,
    calibration_id,
    object_hint=None,
    config=None,
    device=None,
    protected_keep=None,
):
    """Return (same-canvas PIL source, fixed request, semantic preparation record).

    Use the returned source and ORIGINAL instruction in inference.run. This
    auxiliary branch alone encodes the object record. Predictor weights run on
    CPU; Qwen/VAE use the pipeline's model-offload execution device. An explicit
    device, if supplied, must match it.
    Automatic roles are not substituted by manual masks if the checkpoint fails.
    Thresholds/scales/tolerances and calibration_id must come from held-out data;
    the identifier is caller provenance, not an independent verification.
    """
    # Fail before touching large models if only a toy/untrained artifact exists.
    predictor, metadata = load_attribute_checkpoint(
        checkpoint, for_inference=True, device="cpu"
    )
    operation = parse_operation(instruction)
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError(
            "explicit threshold/observer calibration provenance is required"
        )
    if not getattr(pipeline, "enable_model_cpu_offload_flag", False):
        raise ValueError("semantic bridge currently requires model CPU offload")
    actual_device = getattr(pipeline, "_offload_device", None)
    if actual_device is None:
        raise ValueError("missing offload execution device")
    actual_device = torch.device(actual_device)
    device = actual_device if device is None else torch.device(device)
    if device != actual_device:
        raise ValueError("device must match the pipeline offload execution device")
    if not isinstance(source_image, Image.Image):
        raise TypeError("source_image must be a PIL image")
    canvas = (source_image.width // 16 * 16, source_image.height // 16 * 16)
    if min(canvas) < 32:
        raise ValueError("source canvas must be at least 32 pixels per side")
    source = source_image.convert("RGB").resize(canvas, Image.Resampling.LANCZOS)
    _validate_protected_keep(operation, protected_keep, canvas[::-1])
    protected_keep = (
        None
        if protected_keep is None
        else protected_keep.detach().float().cpu().clone()
    )
    normalized = pipeline.image_processor.preprocess(
        source,
        height=canvas[1],
        width=canvas[0],
        max_pixels=canvas[0] * canvas[1],
        max_side_length=max(canvas),
    )
    rgb = normalized.float() / 2 + 0.5
    vae = pipeline.vae
    try:
        encoded = pipeline.encode_instruction(
            [operation.condition_text(instruction)],
            do_classifier_free_guidance=False,
            input_images=[[source]],
            device=device,
            use_rewrite_text_instruction=False,
            truncate_instruction_sequence=False,
        )
        tokens, token_mask = (
            encoded[0].detach().float().cpu(),
            encoded[1].detach().bool().cpu(),
        )
        pipeline.mllm.to("cpu")
        mode = vae.encode(
            normalized.to(device=device, dtype=vae.dtype)
        ).latent_dist.mode()
        if vae.config.shift_factor is not None:
            mode = mode - vae.config.shift_factor
        if vae.config.scaling_factor is not None:
            mode = mode * vae.config.scaling_factor
        mode = mode.detach().float().cpu()
    finally:
        # No graph exists in this static branch. Do not change generators/hooks.
        pipeline.mllm.to("cpu")
        vae.to("cpu")
    hint = (
        torch.zeros(1, 1, *canvas[::-1])
        if object_hint is None
        else object_hint.detach().float().cpu()
    )
    prediction = predictor(
        mode,
        rgb.cpu(),
        hint,
        torch.ones(1, 1, *mode.shape[-2:], dtype=torch.bool),
        tokens,
        token_mask,
    )
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    spec = prediction_constraints(
        prediction,
        rgb.cpu(),
        operation,
        threshold=threshold,
        unknown_max=unknown_max,
        scales=scales,
        tolerances=tolerances,
        provenance=f"semantic_precomputed: checkpoint_sha256={digest}; calibration={calibration_id}",
        protected_keep=protected_keep,
    )
    request = AttributeControlRequest(
        spec, config or ControlConfig(), operation.operation
    )
    record = {
        "semantic_predictor_precomputed": True,
        "checkpoint_sha256": digest,
        "checkpoint_metadata": metadata,
        "operation": asdict(operation),
        "calibration_id": calibration_id,
        "threshold": threshold,
        "unknown_max": unknown_max,
        "active_edit_pixels": int(spec.edit.count_nonzero()),
        "explicit_protected_objects": 0
        if protected_keep is None
        else protected_keep.shape[0],
        "protection_source": "none"
        if protected_keep is None
        else "caller_source_aligned_masks",
        "required_inference_canvas": list(canvas),
        "limits": [
            "single object Chinese templates",
            "no semantic efficacy implied by metadata",
        ],
    }
    return source, request, record
