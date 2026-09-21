"""Pair sampling, augmentation, and isolated training for the PRI predictor."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from .losses import (
    PRILoss,
    PRILossReport,
    normalized_inverse_sqrt_class_weights,
)
from .predictor import SemanticPRIPredictor
from .training_cache import PRITrajectoryRecord

_CHECKPOINT_SCHEMA = "boogu.pri.predictor_checkpoint"
_CHECKPOINT_VERSION = 1
_CHECKPOINT_TENSOR_METADATA = frozenset({"schema", "schema_version", "checkpoint_id"})
_CHECKPOINT_POINTER_SCHEMA = "boogu.pri.predictor_checkpoint_pointer"
_CHECKPOINT_POINTER_VERSION = 1
_CHECKPOINT_ID = re.compile(r"[0-9a-f]{32}")
_CHECKPOINT_POINTER_NAME = "current.json"


def _finite_real(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = normalized < minimum if minimum_inclusive else normalized <= minimum
        if invalid:
            relation = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return normalized


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True, eq=False)
class PRITimePair:
    """Two nearby snapshots from one sample and one independent seed pair."""

    sample_id: str
    source_group_id: str
    split: str
    hbar_c: torch.Tensor
    target: torch.Tensor
    confidence: torch.Tensor
    primary_hint: torch.Tensor
    valid_mask: torch.Tensor
    q_pair: torch.Tensor
    kappa_source: torch.Tensor
    source_latent: torch.Tensor
    current_latents: torch.Tensor
    edit_velocities: torch.Tensor
    preserve_velocities: torch.Tensor
    delta_velocities: torch.Tensor
    progress: torch.Tensor
    steps: torch.Tensor
    seeds: torch.Tensor


@dataclass(frozen=True, eq=False)
class PRIPairBatch:
    """A collated batch retaining the paired-time dimension at index 1."""

    hbar_c: torch.Tensor
    target: torch.Tensor
    confidence: torch.Tensor
    primary_hint: torch.Tensor
    valid_mask: torch.Tensor
    q_pair: torch.Tensor
    kappa_source: torch.Tensor
    source_latent: torch.Tensor
    current_latents: torch.Tensor
    edit_velocities: torch.Tensor
    preserve_velocities: torch.Tensor
    delta_velocities: torch.Tensor
    progress: torch.Tensor
    steps: torch.Tensor
    seeds: torch.Tensor
    sample_ids: tuple[str, ...]
    source_group_ids: tuple[str, ...]
    splits: tuple[str, ...]

    def to(self, device: torch.device | str) -> PRIPairBatch:
        """Move all tensor fields to one device without changing audit IDs."""

        values: dict[str, Any] = {}
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            values[descriptor.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return type(self)(**values)


class InvalidCropError(RuntimeError):
    """Raised when bounded synchronous crop resampling finds no valid crop."""


def sample_neighbor_pair(
    record: PRITrajectoryRecord,
    *,
    max_progress_gap: float = 0.35,
    generator: torch.Generator | None = None,
) -> PRITimePair:
    """Sample ``i != j`` from one trajectory under the registered gap bound."""

    if not isinstance(record, PRITrajectoryRecord):
        raise TypeError("record must be a PRITrajectoryRecord")
    gap = _finite_real(
        "max_progress_gap",
        max_progress_gap,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    rho = record.tensors["rho"]
    candidates = [
        (first, second)
        for first in range(rho.numel())
        for second in range(first + 1, rho.numel())
        if float((rho[first] - rho[second]).abs()) <= gap + 1e-7
    ]
    if not candidates:
        raise ValueError("no distinct nearby cached steps satisfy max_progress_gap")
    selected = int(torch.randint(len(candidates), (1,), generator=generator).item())
    first, second = candidates[selected]
    indices = torch.tensor([first, second], dtype=torch.int64)
    tensors = record.tensors
    metadata = record.metadata
    return PRITimePair(
        sample_id=metadata.sample_id,
        source_group_id=metadata.source_group_id,
        split=metadata.split,
        hbar_c=tensors["hbar_c"].clone(),
        target=tensors["y"].clone(),
        confidence=tensors["w"].clone(),
        primary_hint=tensors["g"].clone(),
        valid_mask=tensors["valid_mask"].clone(),
        q_pair=tensors["q_pair"].clone(),
        kappa_source=tensors["kappa_src"].clone(),
        source_latent=tensors["z_s"].clone(),
        current_latents=tensors["z_k"][indices].unsqueeze(0).clone(),
        edit_velocities=tensors["v_e_k"][indices].unsqueeze(0).clone(),
        preserve_velocities=tensors["v_p_k"][indices].unsqueeze(0).clone(),
        delta_velocities=tensors["delta_v_k"][indices].unsqueeze(0).clone(),
        progress=tensors["rho"][indices].unsqueeze(0).clone(),
        steps=tensors["k"][indices].clone(),
        seeds=tensors["seeds"].clone(),
    )


def _crop_window(
    value: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    *,
    flip: bool,
) -> torch.Tensor:
    cropped = value[..., top : top + height, left : left + width]
    if flip:
        cropped = cropped.flip(-1)
    return cropped.contiguous().clone()


def crop_pri_pair(
    pair: PRITimePair,
    *,
    crop_size: tuple[int, int],
    generator: torch.Generator | None = None,
    tau_mass: float = 1.0,
    max_attempts: int = 16,
    horizontal_flip_probability: float = 0.0,
) -> PRITimePair:
    """Synchronously crop/flip every grid tensor, retrying invalid windows."""

    if not isinstance(pair, PRITimePair):
        raise TypeError("pair must be a PRITimePair")
    if not isinstance(crop_size, tuple) or len(crop_size) != 2:
        raise TypeError("crop_size must be a (height, width) tuple")
    crop_height = _positive_int("crop height", crop_size[0])
    crop_width = _positive_int("crop width", crop_size[1])
    threshold = _finite_real(
        "tau_mass",
        tau_mass,
        minimum=0.0,
        minimum_inclusive=False,
    )
    attempts = _positive_int("max_attempts", max_attempts)
    flip_probability = _finite_real(
        "horizontal_flip_probability",
        horizontal_flip_probability,
        minimum=0.0,
        maximum=1.0,
    )
    full_height, full_width = pair.target.shape[-2:]
    if crop_height > full_height or crop_width > full_width:
        raise ValueError("crop_size cannot exceed the cached latent grid")

    spatial_names = (
        "target",
        "confidence",
        "primary_hint",
        "valid_mask",
        "source_latent",
        "current_latents",
        "edit_velocities",
        "preserve_velocities",
        "delta_velocities",
    )
    for name in spatial_names:
        if getattr(pair, name).shape[-2:] != (full_height, full_width):
            raise ValueError(f"{name} spatial grid must match target before cropping")

    for _ in range(attempts):
        top = int(
            torch.randint(
                full_height - crop_height + 1,
                (1,),
                generator=generator,
            ).item()
        )
        left = int(
            torch.randint(
                full_width - crop_width + 1,
                (1,),
                generator=generator,
            ).item()
        )
        valid_crop = pair.valid_mask[
            ..., top : top + crop_height, left : left + crop_width
        ]
        confidence_crop = pair.confidence[
            ..., top : top + crop_height, left : left + crop_width
        ]
        mass = (confidence_crop.float() * valid_crop).sum(dtype=torch.float32)
        if mass.item() < threshold:
            continue
        flip = bool(
            torch.rand((), generator=generator).item() < flip_probability
        )
        replacements = {
            name: _crop_window(
                getattr(pair, name),
                top,
                left,
                crop_height,
                crop_width,
                flip=flip,
            )
            for name in spatial_names
        }
        return PRITimePair(
            **{
                descriptor.name: replacements.get(
                    descriptor.name,
                    getattr(pair, descriptor.name),
                )
                for descriptor in fields(pair)
            }
        )
    raise InvalidCropError(
        f"no valid crop reached tau_mass after {attempts} bounded attempts"
    )


def collate_pri_pairs(pairs: Sequence[PRITimePair]) -> PRIPairBatch:
    """Collate already-cropped pairs; variable grids must be cropped first."""

    if not pairs:
        raise ValueError("pairs must be non-empty")
    if not all(isinstance(pair, PRITimePair) for pair in pairs):
        raise TypeError("every item must be a PRITimePair")
    tensor_names = (
        "hbar_c",
        "target",
        "confidence",
        "primary_hint",
        "valid_mask",
        "q_pair",
        "kappa_source",
        "source_latent",
        "current_latents",
        "edit_velocities",
        "preserve_velocities",
        "delta_velocities",
        "progress",
    )
    for name in tensor_names:
        shapes = {tuple(getattr(pair, name).shape[1:]) for pair in pairs}
        if len(shapes) != 1:
            raise ValueError(f"{name} shapes differ; apply a fixed synchronous crop")
    return PRIPairBatch(
        **{name: torch.cat([getattr(pair, name) for pair in pairs], dim=0) for name in tensor_names},
        steps=torch.stack([pair.steps for pair in pairs], dim=0),
        seeds=torch.stack([pair.seeds for pair in pairs], dim=0),
        sample_ids=tuple(pair.sample_id for pair in pairs),
        source_group_ids=tuple(pair.source_group_id for pair in pairs),
        splits=tuple(pair.split for pair in pairs),
    )


def compute_frozen_class_weights(
    records: Sequence[PRITrajectoryRecord],
    *,
    frequency_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute class weights once from complete uncropped training records."""

    if not records:
        raise ValueError("records must be non-empty")
    numerator = torch.zeros(3, dtype=torch.float32)
    denominator = torch.zeros((), dtype=torch.float32)
    unique_samples: dict[str, PRITrajectoryRecord] = {}
    for record in records:
        if not isinstance(record, PRITrajectoryRecord):
            raise TypeError("records must contain PRITrajectoryRecord values")
        if record.metadata.split != "train":
            raise ValueError("class weights may only use records from the train split")
        previous = unique_samples.get(record.metadata.sample_id)
        if previous is not None:
            same_audit = (
                previous.metadata.source_group_id == record.metadata.source_group_id
                and previous.metadata.dataset_fingerprint
                == record.metadata.dataset_fingerprint
                and previous.metadata.calibration_fingerprint
                == record.metadata.calibration_fingerprint
                and previous.metadata.criterion_fingerprint
                == record.metadata.criterion_fingerprint
            )
            shared_names = (
                "hbar_c",
                "y",
                "w",
                "g",
                "valid_mask",
                "q_pair",
                "kappa_src",
            )
            same_tensors = all(
                torch.equal(previous.tensors[name], record.tensors[name])
                for name in shared_names
            )
            if not same_audit or not same_tensors:
                raise ValueError(
                    "inconsistent shared labels or audit data for duplicate sample_id"
                )
            # Multiple seed trajectories share one label sample and therefore
            # contribute once to f_q, irrespective of seed count.
            continue
        unique_samples[record.metadata.sample_id] = record
        valid = record.tensors["valid_mask"]
        weight = torch.where(
            valid,
            record.tensors["w"],
            torch.zeros_like(record.tensors["w"]),
        ).float()
        numerator += (record.tensors["y"].float() * weight).sum(dim=(0, 2, 3))
        denominator += weight.sum()
    if denominator.item() <= 0:
        raise ValueError("full uncropped training labels have no valid confidence mass")
    frequencies = numerator / denominator
    return normalized_inverse_sqrt_class_weights(
        frequencies,
        frequency_epsilon=frequency_epsilon,
    ).detach().clone()


def build_predictor_optimizer(
    predictor: SemanticPRIPredictor,
    *,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
) -> torch.optim.AdamW:
    """Build AdamW from only the lightweight predictor's trainable parameters."""

    if not isinstance(predictor, SemanticPRIPredictor):
        raise TypeError("predictor must be a SemanticPRIPredictor")
    learning_rate = _finite_real(
        "learning_rate",
        learning_rate,
        minimum=0.0,
        minimum_inclusive=False,
    )
    weight_decay = _finite_real("weight_decay", weight_decay, minimum=0.0)
    parameters = [parameter for parameter in predictor.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("predictor has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> list[int]:
    return [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]


def _validate_optimizer_scope(
    predictor: SemanticPRIPredictor,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected = {id(parameter) for parameter in predictor.parameters() if parameter.requires_grad}
    actual_list = _optimizer_parameter_ids(optimizer)
    actual = set(actual_list)
    if len(actual) != len(actual_list) or actual != expected:
        raise ValueError(
            "optimizer parameters must contain exactly the predictor's trainable parameters"
        )


class PRIPredictorTrainer:
    """A narrow trainer that cannot update Qwen, VAE, DiT, or cached inputs."""

    def __init__(
        self,
        predictor: SemanticPRIPredictor,
        objective: PRILoss,
        optimizer: torch.optim.Optimizer,
        *,
        hint_dropout_probability: float = 0.5,
        gradient_clip_norm: float | None = 1.0,
    ) -> None:
        if not isinstance(predictor, SemanticPRIPredictor):
            raise TypeError("predictor must be a SemanticPRIPredictor")
        if not isinstance(objective, PRILoss):
            raise TypeError("objective must be a PRILoss")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        _validate_optimizer_scope(predictor, optimizer)
        self.predictor = predictor
        self.objective = objective
        self.optimizer = optimizer
        self.hint_dropout_probability = _finite_real(
            "hint_dropout_probability",
            hint_dropout_probability,
            minimum=0.0,
            maximum=1.0,
        )
        self.gradient_clip_norm = (
            None
            if gradient_clip_norm is None
            else _finite_real(
                "gradient_clip_norm",
                gradient_clip_norm,
                minimum=0.0,
                minimum_inclusive=False,
            )
        )
        self.skipped_steps = 0

    def _predict_pair(
        self,
        batch: PRIPairBatch,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        hint = batch.primary_hint * reliability[:, None, None, None]
        outputs: list[torch.Tensor] = []
        for timepoint in range(2):
            result = self.predictor(
                source_latent_grid=batch.source_latent,
                current_latent=batch.current_latents[:, timepoint],
                delta_velocity=batch.delta_velocities[:, timepoint],
                pooled_instruction=batch.hbar_c,
                progress=batch.progress[:, timepoint],
                primary_hint=hint,
                hint_reliability=reliability,
            )
            outputs.append(result.probabilities)
        return torch.stack(outputs, dim=1)

    def _objective(self, predictions: torch.Tensor, batch: PRIPairBatch) -> PRILossReport:
        return self.objective(
            predictions,
            batch.target,
            batch.confidence,
            batch.valid_mask,
            q_pair=batch.q_pair,
            kappa_source=batch.kappa_source,
        )

    def train_step(
        self,
        batch: PRIPairBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> PRILossReport:
        """Run one update with a single shared hint decision per pair."""

        if not isinstance(batch, PRIPairBatch):
            raise TypeError("batch must be a PRIPairBatch")
        if not batch.splits or any(split != "train" for split in batch.splits):
            raise ValueError("train_step requires every sample to come from the train split")
        _validate_optimizer_scope(self.predictor, self.optimizer)
        self.predictor.train()
        self.optimizer.zero_grad(set_to_none=True)
        valid_confidence = torch.where(
            batch.valid_mask,
            batch.confidence.detach().float(),
            torch.zeros_like(batch.confidence, dtype=torch.float32),
        )
        sample_mass = valid_confidence.sum(dim=(1, 2, 3), dtype=torch.float32)
        if (
            torch.isfinite(valid_confidence[batch.valid_mask]).all().item()
            and not (valid_confidence[batch.valid_mask] < 0).any().item()
            and not (sample_mass >= self.objective.config.tau_mass).any().item()
        ):
            neutral = torch.full(
                (
                    batch.target.shape[0],
                    2,
                    3,
                    batch.target.shape[-2],
                    batch.target.shape[-1],
                ),
                1.0 / 3.0,
                device=batch.target.device,
                dtype=torch.float32,
            )
            report = self._objective(neutral, batch)
            if not report.skipped:
                raise RuntimeError("loss validity precheck disagreed with PRILoss")
            self.skipped_steps += 1
            return report
        keep_cpu = (
            torch.rand(batch.target.shape[0], generator=generator)
            >= self.hint_dropout_probability
        )
        reliability = keep_cpu.to(device=batch.target.device, dtype=torch.float32)
        predictions = self._predict_pair(batch, reliability)
        report = self._objective(predictions, batch)
        if report.skipped:
            self.skipped_steps += 1
            self.optimizer.zero_grad(set_to_none=True)
            return report
        assert report.total is not None
        report.total.backward()
        if self.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.predictor.parameters(),
                self.gradient_clip_norm,
            )
        self.optimizer.step()
        return report

    @torch.no_grad()
    def evaluate_step(
        self,
        batch: PRIPairBatch,
        *,
        use_primary_hint: bool = False,
    ) -> PRILossReport:
        """Evaluate deterministically; no-hint is the registered main mode."""

        if not isinstance(batch, PRIPairBatch):
            raise TypeError("batch must be a PRIPairBatch")
        if not batch.splits or any(split != "validation" for split in batch.splits):
            raise ValueError(
                "evaluate_step requires every sample to come from the validation split"
            )
        self.predictor.eval()
        reliability = torch.full(
            (batch.target.shape[0],),
            1.0 if use_primary_hint else 0.0,
            device=batch.target.device,
            dtype=torch.float32,
        )
        return self._objective(self._predict_pair(batch, reliability), batch)


def _normalize_json(value: Any, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} JSON number must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} JSON object keys must be strings")
            result[key] = _normalize_json(item, f"{path}.{key}")
        return result
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _atomic_replace_file(temporary: Path, destination: Path) -> None:
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _predictor_architecture(predictor: SemanticPRIPredictor) -> dict[str, Any]:
    return {
        "class": type(predictor).__name__,
        "latent_channels": predictor.latent_channels,
        "instruction_dim": predictor.instruction_dim,
        "hidden_channels": predictor.hidden_channels,
        "num_groups": predictor.num_groups,
        "probability_parameterization": "hierarchical_stick_breaking",
        "channel_order": ["P", "R", "I"],
    }


def save_predictor_checkpoint(
    predictor: SemanticPRIPredictor,
    output_directory: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Atomically save tensor-only weights plus auditable JSON metadata."""

    if not isinstance(predictor, SemanticPRIPredictor):
        raise TypeError("predictor must be a SemanticPRIPredictor")
    user_metadata = _normalize_json({} if metadata is None else metadata)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    pointer_path = destination / _CHECKPOINT_POINTER_NAME
    if not overwrite and pointer_path.exists():
        raise FileExistsError(f"checkpoint already exists in {destination}")
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in predictor.state_dict().items()
    }
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("predictor state_dict must contain only tensors")
    for name, value in state.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all().item():
            raise ValueError(f"predictor checkpoint tensor {name} must be finite")
    checkpoint_id = uuid4().hex
    generation_directory = destination / "generations" / checkpoint_id
    weights_path = generation_directory / "predictor.safetensors"
    metadata_path = generation_directory / "metadata.json"
    document = {
        "schema": _CHECKPOINT_SCHEMA,
        "schema_version": _CHECKPOINT_VERSION,
        "checkpoint_id": checkpoint_id,
        "architecture": _predictor_architecture(predictor),
        "state_keys": sorted(state),
        "user_metadata": user_metadata,
    }
    pointer_document = {
        "schema": _CHECKPOINT_POINTER_SCHEMA,
        "schema_version": _CHECKPOINT_POINTER_VERSION,
        "checkpoint_id": checkpoint_id,
    }
    pointer_text = (
        json.dumps(
            pointer_document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    temp_pointer = destination / f".{_CHECKPOINT_POINTER_NAME}.{checkpoint_id}.tmp"
    generation_directory.mkdir(parents=True, exist_ok=False)
    committed = False
    try:
        save_file(
            state,
            str(weights_path),
            metadata={
                "schema": _CHECKPOINT_SCHEMA,
                "schema_version": str(_CHECKPOINT_VERSION),
                "checkpoint_id": checkpoint_id,
            },
        )
        metadata_path.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        with weights_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        with metadata_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        temp_pointer.write_text(pointer_text, encoding="utf-8")
        # The immutable generation is complete before this single commit point.
        # A failed overwrite therefore leaves the previous pointer loadable.
        _atomic_replace_file(temp_pointer, pointer_path)
        committed = True
    finally:
        if temp_pointer.exists():
            temp_pointer.unlink()
        if not committed:
            # ``os.replace`` is the sole commit point.  A wrapper could report an
            # error after the replacement itself succeeded, so compare the exact
            # pointer bytes before deciding that this generation is uncommitted.
            try:
                committed = pointer_path.read_text(encoding="utf-8") == pointer_text
            except (OSError, UnicodeError):
                committed = False
        if not committed:
            for incomplete_path in (weights_path, metadata_path):
                try:
                    incomplete_path.unlink()
                except FileNotFoundError:
                    pass
            try:
                generation_directory.rmdir()
            except OSError:
                # Never recurse through an unexpected directory layout here.
                pass
    return weights_path, metadata_path


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as error:
        raise ValueError("checkpoint metadata is not valid JSON") from error
    if not isinstance(document, Mapping):
        raise TypeError("checkpoint metadata must contain one JSON object")
    return document


def load_predictor_checkpoint(
    predictor: SemanticPRIPredictor,
    checkpoint_directory: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Strictly load a safe predictor checkpoint; no pickle fallback exists."""

    if not isinstance(predictor, SemanticPRIPredictor):
        raise TypeError("predictor must be a SemanticPRIPredictor")
    source = Path(checkpoint_directory)
    pointer_path = source / _CHECKPOINT_POINTER_NAME
    pointer = _load_strict_json(pointer_path)
    if set(pointer) != {"schema", "schema_version", "checkpoint_id"}:
        raise ValueError("checkpoint pointer contains missing or unknown keys")
    if pointer["schema"] != _CHECKPOINT_POINTER_SCHEMA:
        raise ValueError("unsupported predictor checkpoint pointer schema")
    if isinstance(pointer["schema_version"], bool) or not isinstance(
        pointer["schema_version"], Integral
    ):
        raise TypeError("checkpoint pointer schema_version must be an integer")
    if int(pointer["schema_version"]) != _CHECKPOINT_POINTER_VERSION:
        raise ValueError("unsupported predictor checkpoint pointer version")
    checkpoint_id = pointer["checkpoint_id"]
    if not isinstance(checkpoint_id, str) or _CHECKPOINT_ID.fullmatch(checkpoint_id) is None:
        raise ValueError("checkpoint pointer checkpoint_id must be a lowercase UUID hex")
    generation_directory = source / "generations" / checkpoint_id
    weights_path = generation_directory / "predictor.safetensors"
    metadata_path = generation_directory / "metadata.json"
    document = _load_strict_json(metadata_path)
    expected_document_keys = {
        "schema",
        "schema_version",
        "checkpoint_id",
        "architecture",
        "state_keys",
        "user_metadata",
    }
    if set(document) != expected_document_keys:
        raise ValueError("checkpoint JSON contains missing or unknown keys")
    if not isinstance(document["schema"], str):
        raise TypeError("checkpoint schema must be a JSON string")
    if isinstance(document["schema_version"], bool) or not isinstance(
        document["schema_version"], Integral
    ):
        raise TypeError("checkpoint schema_version must be an integer")
    if not isinstance(document["checkpoint_id"], str) or not document[
        "checkpoint_id"
    ]:
        raise TypeError("checkpoint_id must be a non-empty JSON string")
    if not isinstance(document["architecture"], Mapping):
        raise TypeError("checkpoint architecture must be a JSON object")
    if not isinstance(document["state_keys"], list) or not all(
        isinstance(name, str) for name in document["state_keys"]
    ):
        raise TypeError("checkpoint state_keys must be a JSON string list")
    if not isinstance(document["user_metadata"], Mapping):
        raise TypeError("checkpoint user_metadata must be a JSON object")
    if (
        document["schema"] != _CHECKPOINT_SCHEMA
        or int(document["schema_version"]) != _CHECKPOINT_VERSION
    ):
        raise ValueError("unsupported predictor checkpoint schema/version")
    if document["architecture"] != _predictor_architecture(predictor):
        raise ValueError("checkpoint predictor architecture does not match target")
    try:
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            tensor_metadata = handle.metadata()
            tensor_keys = set(handle.keys())
            if not isinstance(tensor_metadata, Mapping):
                raise TypeError("predictor safetensors checkpoint is missing metadata")
            if set(tensor_metadata) != _CHECKPOINT_TENSOR_METADATA:
                raise ValueError("predictor safetensors metadata is missing or unknown")
            if (
                tensor_metadata["schema"] != _CHECKPOINT_SCHEMA
                or tensor_metadata["schema_version"] != str(_CHECKPOINT_VERSION)
                or tensor_metadata["checkpoint_id"] != document["checkpoint_id"]
                or tensor_metadata["checkpoint_id"] != checkpoint_id
            ):
                raise ValueError("predictor checkpoint files do not belong to the same save")
            expected_keys = set(predictor.state_dict())
            if tensor_keys != expected_keys or document["state_keys"] != sorted(expected_keys):
                raise ValueError("predictor checkpoint state keys do not match exactly")
            state = {name: handle.get_tensor(name) for name in tensor_keys}
    except (TypeError, ValueError):
        raise
    except Exception as error:
        raise ValueError("invalid predictor safetensors checkpoint") from error
    target_state = predictor.state_dict()
    for name, value in state.items():
        expected = target_state[name]
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(f"predictor checkpoint tensor {name} has wrong shape")
        if value.dtype != expected.dtype:
            raise ValueError(f"predictor checkpoint tensor {name} has wrong dtype")
        if torch.is_floating_point(value) and not torch.isfinite(value).all().item():
            raise ValueError(f"predictor checkpoint tensor {name} must be finite")
    try:
        predictor.load_state_dict(state, strict=True)
    except Exception as error:
        raise ValueError("predictor checkpoint failed after full validation") from error
    return document


def load_predictor_from_checkpoint(
    checkpoint_directory: str | os.PathLike[str],
) -> tuple[SemanticPRIPredictor, Mapping[str, Any]]:
    """Construct and strictly load the registered predictor architecture."""

    source = Path(checkpoint_directory)
    pointer = _load_strict_json(source / _CHECKPOINT_POINTER_NAME)
    if set(pointer) != {"schema", "schema_version", "checkpoint_id"}:
        raise ValueError("checkpoint pointer contains missing or unknown keys")
    if (
        pointer["schema"] != _CHECKPOINT_POINTER_SCHEMA
        or isinstance(pointer["schema_version"], bool)
        or not isinstance(pointer["schema_version"], Integral)
        or int(pointer["schema_version"]) != _CHECKPOINT_POINTER_VERSION
    ):
        raise ValueError("unsupported predictor checkpoint pointer schema/version")
    checkpoint_id = pointer["checkpoint_id"]
    if not isinstance(checkpoint_id, str) or _CHECKPOINT_ID.fullmatch(checkpoint_id) is None:
        raise ValueError("checkpoint pointer checkpoint_id must be a lowercase UUID hex")

    document = _load_strict_json(
        source / "generations" / checkpoint_id / "metadata.json"
    )
    architecture = document.get("architecture")
    expected_architecture_keys = {
        "class",
        "latent_channels",
        "instruction_dim",
        "hidden_channels",
        "num_groups",
        "probability_parameterization",
        "channel_order",
    }
    if not isinstance(architecture, Mapping) or set(architecture) != expected_architecture_keys:
        raise ValueError("checkpoint predictor architecture is missing or unknown")
    if architecture["class"] != "SemanticPRIPredictor":
        raise ValueError("unsupported predictor class")
    if architecture["probability_parameterization"] != "hierarchical_stick_breaking":
        raise ValueError("unsupported predictor probability parameterization")
    if architecture["channel_order"] != ["P", "R", "I"]:
        raise ValueError("unsupported predictor channel order")

    latent_channels = _positive_int("latent_channels", architecture["latent_channels"])
    instruction_dim = _positive_int("instruction_dim", architecture["instruction_dim"])
    hidden_channels = _positive_int("hidden_channels", architecture["hidden_channels"])
    num_groups = _positive_int("num_groups", architecture["num_groups"])
    if latent_channels > 64 or instruction_dim > 8192 or hidden_channels > 128:
        raise ValueError("predictor checkpoint architecture exceeds safe inference bounds")

    predictor = SemanticPRIPredictor(
        latent_channels=latent_channels,
        instruction_dim=instruction_dim,
        hidden_channels=hidden_channels,
        num_groups=num_groups,
    )
    parameter_count = sum(parameter.numel() for parameter in predictor.parameters())
    if parameter_count >= 500_000:
        raise ValueError("PRI predictor checkpoint exceeds the 0.5M parameter budget")
    loaded_document = load_predictor_checkpoint(predictor, source)
    predictor.requires_grad_(False)
    predictor.eval()
    return predictor, loaded_document


__all__ = [
    "InvalidCropError",
    "PRIPairBatch",
    "PRIPredictorTrainer",
    "PRITimePair",
    "build_predictor_optimizer",
    "collate_pri_pairs",
    "compute_frozen_class_weights",
    "crop_pri_pair",
    "load_predictor_checkpoint",
    "load_predictor_from_checkpoint",
    "sample_neighbor_pair",
    "save_predictor_checkpoint",
]
