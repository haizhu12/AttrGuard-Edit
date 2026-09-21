"""Safe, auditable storage for teacher-forced PRI training trajectories.

One cache file represents one source sample, one independent reference/noise
seed pair, and all pre-registered denoising snapshots for that trajectory.  The
format intentionally has no pickle fallback: tensors are stored by
``safetensors`` and the only metadata payload is a strictly validated JSON
object containing scalar values and flat scalar lists.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PRI_CACHE_SCHEMA = "boogu.pri.teacher_trajectory"
PRI_CACHE_VERSION = 2
CACHE_METADATA_KEY = "boogu_pri_metadata"

_MANIFEST_SCHEMA = "boogu.pri.cache_manifest"
_MANIFEST_VERSION = 2
_PRODUCER_VERSION = "boogu-pri-phase1-v1"
_CURRENT_SCHEMA = "boogu.pri.cache_current"
_CURRENT_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPLITS = frozenset({"train", "validation", "test"})
_FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
_TENSOR_KEYS = frozenset(
    {
        "hbar_c",
        "y",
        "w",
        "g",
        "valid_mask",
        "q_pair",
        "kappa_src",
        "z_s",
        "z_k",
        "v_e_k",
        "v_p_k",
        "delta_v_k",
        "rho",
        "k",
        "seeds",
    }
)


def _strict_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if not -(2**63) <= normalized < 2**63:
        raise ValueError(f"{name} must fit in a signed 64-bit integer")
    return normalized


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a JSON string scalar")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string")
    return value


def _fingerprint(name: str, value: Any) -> str:
    normalized = _nonempty_string(name, value)
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest")
    return normalized


def _flat_json_value(name: str, value: Any) -> None:
    scalars = (str, int, float, bool, type(None))
    if isinstance(value, scalars) and not isinstance(value, complex):
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)).item():
            raise ValueError(f"{name} must be finite")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if not isinstance(item, scalars) or isinstance(item, complex):
                raise TypeError(
                    f"{name}[{index}] must be a JSON scalar; nested objects are forbidden"
                )
            if isinstance(item, float) and not torch.isfinite(torch.tensor(item)).item():
                raise ValueError(f"{name}[{index}] must be finite")
        return
    raise TypeError(f"{name} must be a JSON scalar or a flat list of JSON scalars")


@dataclass(frozen=True)
class PRICacheMetadata:
    """Strict audit metadata for one cached sample/seed trajectory."""

    sample_id: str
    source_group_id: str
    split: str
    dataset_fingerprint: str
    calibration_fingerprint: str
    criterion_name: str
    criterion_version: str
    criterion_fingerprint: str
    teacher_config_fingerprint: str
    num_inference_steps: int
    selected_steps: tuple[int, ...]
    seed_ref: int
    seed_noise: int
    latent_channels: int
    latent_height: int
    latent_width: int
    schema: str = PRI_CACHE_SCHEMA
    schema_version: int = PRI_CACHE_VERSION
    producer_version: str = "boogu-pri-phase1-v1"
    conditioning_representation: str = "masked_mean_pre_projection"
    channel_order: tuple[str, ...] = ("P", "R", "I")
    tensor_layout: tuple[str, ...] = (
        "shared:BCHW",
        "trajectory:KCHW",
        "probability:P_R_I",
    )
    trajectory_mode: str = "teacher_forced_baseline"
    delta_convention: str = "v_e_minus_v_p"
    step_indexing: str = "zero_based"
    progress_formula: str = "k_over_n_minus_1"
    seed_names: tuple[str, ...] = ("seed_ref", "seed_noise")
    rng_streams_independent: bool = True

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            _flat_json_value(descriptor.name, getattr(self, descriptor.name))

        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, Integral
        ):
            raise TypeError("schema_version must be an integer, not a boolean")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if not isinstance(self.rng_streams_independent, bool):
            raise TypeError("rng_streams_independent must be a JSON boolean")

        object.__setattr__(self, "sample_id", _nonempty_string("sample_id", self.sample_id))
        object.__setattr__(
            self,
            "source_group_id",
            _nonempty_string("source_group_id", self.source_group_id),
        )
        if self.split not in _SPLITS:
            raise ValueError("split must be exactly one of train, validation, or test")
        for name in (
            "dataset_fingerprint",
            "calibration_fingerprint",
            "criterion_fingerprint",
            "teacher_config_fingerprint",
        ):
            object.__setattr__(self, name, _fingerprint(name, getattr(self, name)))
        object.__setattr__(
            self,
            "criterion_name",
            _nonempty_string("criterion_name", self.criterion_name),
        )
        object.__setattr__(
            self,
            "criterion_version",
            _nonempty_string("criterion_version", self.criterion_version),
        )
        object.__setattr__(
            self,
            "producer_version",
            _nonempty_string("producer_version", self.producer_version),
        )
        steps = _strict_int("num_inference_steps", self.num_inference_steps, minimum=2)
        object.__setattr__(self, "num_inference_steps", steps)
        if not isinstance(self.selected_steps, (tuple, list)):
            raise TypeError("selected_steps must be a flat integer list")
        selected = tuple(
            _strict_int(f"selected_steps[{index}]", value, minimum=0)
            for index, value in enumerate(self.selected_steps)
        )
        if len(selected) < 2:
            raise ValueError("selected_steps must contain at least two steps")
        if selected != tuple(sorted(selected)):
            raise ValueError("selected_steps must be sorted in ascending order")
        if len(set(selected)) != len(selected):
            raise ValueError("selected_steps must be unique")
        if selected[-1] >= steps:
            raise ValueError("selected_steps must be zero-based indices below N")
        object.__setattr__(self, "selected_steps", selected)
        for name in ("seed_ref", "seed_noise"):
            object.__setattr__(
                self,
                name,
                _strict_int(name, getattr(self, name), minimum=0),
            )
        if self.seed_ref == self.seed_noise:
            raise ValueError("seed_ref and seed_noise must be distinct independent streams")
        for name in ("latent_channels", "latent_height", "latent_width"):
            object.__setattr__(
                self,
                name,
                _strict_int(name, getattr(self, name), minimum=1),
            )

        exact_values = {
            "schema": PRI_CACHE_SCHEMA,
            "schema_version": PRI_CACHE_VERSION,
            "conditioning_representation": "masked_mean_pre_projection",
            "channel_order": ("P", "R", "I"),
            "tensor_layout": (
                "shared:BCHW",
                "trajectory:KCHW",
                "probability:P_R_I",
            ),
            "trajectory_mode": "teacher_forced_baseline",
            "delta_convention": "v_e_minus_v_p",
            "step_indexing": "zero_based",
            "progress_formula": "k_over_n_minus_1",
            "seed_names": ("seed_ref", "seed_noise"),
            "rng_streams_independent": True,
        }
        for name, expected in exact_values.items():
            value = getattr(self, name)
            if isinstance(expected, tuple):
                value = tuple(value)
            if value != expected:
                raise ValueError(f"{name} must equal the registered cache convention")
            if isinstance(expected, tuple):
                object.__setattr__(self, name, expected)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical flat JSON representation."""

        result: dict[str, Any] = {}
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            result[descriptor.name] = list(value) if isinstance(value, tuple) else value
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> PRICacheMetadata:
        if not isinstance(values, Mapping):
            raise TypeError("cache metadata must be a JSON object")
        expected = {descriptor.name for descriptor in fields(cls)}
        provided = set(values)
        unknown = provided - expected
        missing = expected - provided
        if unknown:
            raise ValueError(f"unknown metadata keys: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing metadata keys: {sorted(missing)}")
        normalized = dict(values)
        for name in ("selected_steps", "channel_order", "tensor_layout", "seed_names"):
            if not isinstance(normalized[name], list):
                raise TypeError(f"{name} must be a JSON list")
            normalized[name] = tuple(normalized[name])
        return cls(**normalized)


def _validate_exact_tensor_keys(tensors: Mapping[str, Any]) -> None:
    if not isinstance(tensors, Mapping):
        raise TypeError("tensors must be a mapping")
    provided = set(tensors)
    unknown = provided - _TENSOR_KEYS
    missing = _TENSOR_KEYS - provided
    if unknown:
        raise ValueError(f"unknown tensor keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing tensor keys: {sorted(missing)}")


def _require_tensor(
    tensors: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype | frozenset[torch.dtype],
) -> torch.Tensor:
    value = tensors[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    accepted = dtype if isinstance(dtype, frozenset) else frozenset({dtype})
    if value.dtype not in accepted:
        expected = "FP32" if accepted == frozenset({torch.float32}) else str(accepted)
        raise TypeError(f"{name} must use {expected} dtype")
    if torch.is_floating_point(value) and not torch.isfinite(value.detach()).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_unit_interval(name: str, value: torch.Tensor, *, positive: bool = False) -> None:
    detached = value.detach()
    lower_bad = detached <= 0 if positive else detached < 0
    if lower_bad.any().item() or (detached > 1).any().item():
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} values must lie in {interval}")


def _validated_tensor_clones(
    tensors: Mapping[str, Any],
    metadata: PRICacheMetadata,
) -> dict[str, torch.Tensor]:
    _validate_exact_tensor_keys(tensors)
    height = metadata.latent_height
    width = metadata.latent_width
    channels = metadata.latent_channels
    count = len(metadata.selected_steps)

    _require_tensor(tensors, "hbar_c", (1, 4096), torch.float32)
    labels = _require_tensor(tensors, "y", (1, 3, height, width), torch.float32)
    confidence = _require_tensor(tensors, "w", (1, 1, height, width), torch.float32)
    hint = _require_tensor(tensors, "g", (1, 1, height, width), torch.float32)
    valid = _require_tensor(tensors, "valid_mask", (1, 1, height, width), torch.bool)
    q_pair = _require_tensor(tensors, "q_pair", (1,), torch.float32)
    kappa = _require_tensor(tensors, "kappa_src", (1,), torch.float32)
    source = _require_tensor(tensors, "z_s", (1, channels, height, width), _FLOAT_DTYPES)
    trajectory_shape = (count, channels, height, width)
    current = _require_tensor(tensors, "z_k", trajectory_shape, _FLOAT_DTYPES)
    edit = _require_tensor(tensors, "v_e_k", trajectory_shape, _FLOAT_DTYPES)
    preserve = _require_tensor(tensors, "v_p_k", trajectory_shape, _FLOAT_DTYPES)
    residual = _require_tensor(tensors, "delta_v_k", trajectory_shape, _FLOAT_DTYPES)
    progress = _require_tensor(tensors, "rho", (count,), torch.float32)
    step_indices = _require_tensor(tensors, "k", (count,), torch.int64)
    seeds = _require_tensor(tensors, "seeds", (2,), torch.int64)

    for name, value in (("y", labels), ("w", confidence), ("g", hint)):
        _validate_unit_interval(name, value)
    label_sum = labels.detach().sum(dim=1)
    if not torch.allclose(label_sum, torch.ones_like(label_sum), rtol=0.0, atol=1e-4):
        raise ValueError("y probabilities must sum to 1 in P/R/I channel order")
    _validate_unit_interval("q_pair", q_pair)
    _validate_unit_interval("kappa_src", kappa, positive=True)
    if not valid.any().item():
        raise ValueError("valid_mask must contain at least one valid position")
    if (confidence.detach() * valid).sum().item() <= 0:
        raise ValueError("w must have positive mass on valid positions")

    if source.dtype != current.dtype:
        raise TypeError("z_s and z_k must use the same dtype")
    if not (edit.dtype == preserve.dtype == residual.dtype):
        raise TypeError("v_e_k, v_p_k, and delta_v_k must use the same dtype")
    expected_residual = edit.detach().float() - preserve.detach().float()
    tolerance = 5e-3 if edit.dtype in (torch.float16, torch.bfloat16) else 1e-5
    if not torch.allclose(
        residual.detach().float(),
        expected_residual,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("delta_v_k must equal v_e_k - v_p_k within dtype tolerance")

    expected_indices = torch.tensor(metadata.selected_steps, dtype=torch.int64)
    if not torch.equal(step_indices.detach().cpu(), expected_indices):
        if tuple(step_indices.detach().cpu().tolist()) != tuple(
            sorted(step_indices.detach().cpu().tolist())
        ):
            raise ValueError("k indices must be sorted, unique, zero-based selected steps")
        raise ValueError("k indices must exactly match metadata selected_steps")
    expected_progress = expected_indices.float() / float(metadata.num_inference_steps - 1)
    if not torch.allclose(
        progress.detach().cpu(),
        expected_progress,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("rho must equal k / (N - 1) for the registered trajectory")
    expected_seeds = torch.tensor(
        [metadata.seed_ref, metadata.seed_noise],
        dtype=torch.int64,
    )
    if not torch.equal(seeds.detach().cpu(), expected_seeds):
        raise ValueError("seeds tensor must match seed_ref/seed_noise metadata")

    # Copy at the trust boundary.  Cache payloads never retain caller storage,
    # autograd graphs, devices, or non-contiguous views.
    return {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in tensors.items()
    }


@dataclass(frozen=True, eq=False)
class PRITrajectoryRecord:
    """Validated immutable mapping shell around one cache payload."""

    tensors: Mapping[str, torch.Tensor]
    metadata: PRICacheMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PRICacheMetadata):
            raise TypeError("metadata must be a PRICacheMetadata")
        clones = _validated_tensor_clones(self.tensors, self.metadata)
        object.__setattr__(self, "tensors", MappingProxyType(clones))


def _coerce_metadata(metadata: PRICacheMetadata | Mapping[str, Any]) -> PRICacheMetadata:
    if isinstance(metadata, PRICacheMetadata):
        return metadata
    return PRICacheMetadata.from_dict(metadata)


def _ensure_safetensors_path(path: str | os.PathLike[str]) -> Path:
    normalized = Path(path)
    if normalized.suffix.lower() != ".safetensors":
        raise ValueError("PRI cache paths must use the .safetensors extension")
    return normalized


def _canonical_metadata_json(metadata: PRICacheMetadata) -> str:
    return json.dumps(
        metadata.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_pri_cache(
    path: str | os.PathLike[str],
    tensors: Mapping[str, Any],
    metadata: PRICacheMetadata | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically write one safe trajectory cache file."""

    destination = _ensure_safetensors_path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"cache already exists: {destination}")
    record = PRITrajectoryRecord(tensors, _coerce_metadata(metadata))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    try:
        save_file(
            dict(record.tensors),
            str(temporary),
            metadata={CACHE_METADATA_KEY: _canonical_metadata_json(record.metadata)},
        )
        # Windows requires a writable descriptor for ``FlushFileBuffers`` via
        # ``os.fsync`` even though no additional bytes are written here.
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _strict_json_object(payload: str) -> Mapping[str, Any]:
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
        parsed = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as error:
        raise ValueError("cache metadata is not valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise TypeError("cache metadata JSON must contain one object")
    return parsed


def read_pri_cache(path: str | os.PathLike[str]) -> PRITrajectoryRecord:
    """Load one cache file and revalidate every tensor and audit convention."""

    source = _ensure_safetensors_path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            raw_metadata = handle.metadata()
            tensor_keys = set(handle.keys())
            if not isinstance(raw_metadata, Mapping):
                raise TypeError("cache is missing the registered metadata mapping")
            if set(raw_metadata) != {CACHE_METADATA_KEY}:
                raise ValueError("cache must contain exactly the registered metadata key")
            _validate_exact_tensor_keys({key: None for key in tensor_keys})
            metadata = PRICacheMetadata.from_dict(
                _strict_json_object(raw_metadata[CACHE_METADATA_KEY])
            )
            tensors = {key: handle.get_tensor(key) for key in tensor_keys}
    except (TypeError, ValueError):
        raise
    except Exception as error:
        raise ValueError(f"invalid safetensors cache: {source}") from error
    return PRITrajectoryRecord(tensors, metadata)


@dataclass(frozen=True)
class PRICacheIndexEntry:
    """Lightweight cache header used to plan streaming reads."""

    path: Path
    metadata: PRICacheMetadata


def index_pri_cache(path: str | os.PathLike[str]) -> PRICacheIndexEntry:
    """Validate one cache header without decoding any tensor payload."""

    source = _ensure_safetensors_path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            raw_metadata = handle.metadata()
            tensor_keys = set(handle.keys())
            if not isinstance(raw_metadata, Mapping):
                raise TypeError("cache is missing the registered metadata mapping")
            if set(raw_metadata) != {CACHE_METADATA_KEY}:
                raise ValueError("cache must contain exactly the registered metadata key")
            _validate_exact_tensor_keys({key: None for key in tensor_keys})
            metadata = PRICacheMetadata.from_dict(
                _strict_json_object(raw_metadata[CACHE_METADATA_KEY])
            )
    except (TypeError, ValueError):
        raise
    except Exception as error:
        raise ValueError(f"invalid safetensors cache: {source}") from error
    return PRICacheIndexEntry(path=source.resolve(), metadata=metadata)


@dataclass(frozen=True)
class PRICacheProtocol:
    producer_version: str
    teacher_config_fingerprint: str
    num_inference_steps: int
    selected_steps: tuple[int, ...]
    seed_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PRICacheManifest:
    train: tuple[Path, ...]
    validation: tuple[Path, ...]
    protocol: PRICacheProtocol


def _resolve_current_manifest(source: Path, document: Mapping[str, Any]) -> Path:
    expected = {
        "schema",
        "schema_version",
        "generation",
        "training_manifest",
    }
    if set(document) != expected:
        raise ValueError("current pointer must contain exactly the registered keys")
    if document["schema"] != _CURRENT_SCHEMA:
        raise ValueError("unsupported cache current-pointer schema")
    if (
        isinstance(document["schema_version"], bool)
        or not isinstance(document["schema_version"], Integral)
        or int(document["schema_version"]) != _CURRENT_VERSION
    ):
        raise ValueError("unsupported cache current-pointer schema version")
    root = source.resolve().parent
    generation_text = _nonempty_string("current generation", document["generation"])
    manifest_text = _nonempty_string(
        "current training_manifest", document["training_manifest"]
    )
    generation_relative = Path(generation_text)
    manifest_relative = Path(manifest_text)
    if generation_relative.is_absolute() or manifest_relative.is_absolute():
        raise ValueError("current pointer paths must be relative to the cache root")
    try:
        generations_root = (root / "generations").resolve(strict=True)
        generation = (root / generation_relative).resolve(strict=True)
        manifest = (root / manifest_relative).resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError("current pointer references a missing generation") from error
    if generation.parent != generations_root or not generation.is_dir():
        raise ValueError("current pointer generation must be a direct generation directory")
    if manifest.parent != generation or manifest.name != "cache_manifest.json":
        raise ValueError("current pointer training manifest must belong to its generation")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return manifest


def load_cache_manifest(path: str | os.PathLike[str]) -> PRICacheManifest:
    """Read a strict manifest or resolve an atomically committed cache pointer."""

    source = Path(path)
    if source.is_dir():
        source = source / "current.json"
    try:
        document = _strict_json_object(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(source) from error
    if document.get("schema") == _CURRENT_SCHEMA:
        source = _resolve_current_manifest(source, document)
        try:
            document = _strict_json_object(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise FileNotFoundError(source) from error
    expected = {
        "schema",
        "schema_version",
        "producer_version",
        "num_inference_steps",
        "selected_steps",
        "seed_pairs",
        "teacher_config_fingerprint",
        "train",
        "validation",
    }
    unknown = set(document) - expected
    missing = expected - set(document)
    if unknown:
        raise ValueError(f"unknown manifest keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing manifest keys: {sorted(missing)}")
    if document["schema"] != _MANIFEST_SCHEMA:
        raise ValueError("unsupported cache manifest schema")
    if (
        isinstance(document["schema_version"], bool)
        or not isinstance(document["schema_version"], Integral)
        or int(document["schema_version"]) != _MANIFEST_VERSION
    ):
        raise ValueError("unsupported cache manifest schema version")
    producer_version = _nonempty_string(
        "manifest producer_version",
        document["producer_version"],
    )
    if producer_version != _PRODUCER_VERSION:
        raise ValueError("unsupported cache manifest producer_version")
    teacher_config_fingerprint = _fingerprint(
        "manifest teacher_config_fingerprint",
        document["teacher_config_fingerprint"],
    )
    num_inference_steps = _strict_int(
        "manifest num_inference_steps",
        document["num_inference_steps"],
        minimum=2,
    )
    selected_raw = document["selected_steps"]
    if not isinstance(selected_raw, list):
        raise TypeError("manifest selected_steps must be a JSON list")
    selected_steps = tuple(
        _strict_int(f"manifest selected_steps[{index}]", value, minimum=0)
        for index, value in enumerate(selected_raw)
    )
    if len(selected_steps) < 2:
        raise ValueError("manifest selected_steps must contain at least two steps")
    if selected_steps != tuple(sorted(set(selected_steps))):
        raise ValueError("manifest selected_steps must be sorted and unique")
    if selected_steps[-1] >= num_inference_steps:
        raise ValueError("manifest selected_steps must be below num_inference_steps")
    seed_pairs_raw = document["seed_pairs"]
    if not isinstance(seed_pairs_raw, list) or not seed_pairs_raw:
        raise ValueError("manifest seed_pairs must be a non-empty JSON list")
    seed_pairs_list: list[tuple[int, int]] = []
    for index, pair in enumerate(seed_pairs_raw):
        if not isinstance(pair, Mapping):
            raise TypeError(f"manifest seed_pairs[{index}] must be a JSON object")
        if set(pair) != {"seed_ref", "seed_noise"}:
            raise ValueError(
                f"manifest seed_pairs[{index}] must contain exactly seed_ref and seed_noise"
            )
        seed_ref = _strict_int(
            f"manifest seed_pairs[{index}].seed_ref",
            pair["seed_ref"],
            minimum=0,
        )
        seed_noise = _strict_int(
            f"manifest seed_pairs[{index}].seed_noise",
            pair["seed_noise"],
            minimum=0,
        )
        if seed_ref == seed_noise:
            raise ValueError(
                f"manifest seed_pairs[{index}] must use distinct independent seeds"
            )
        seed_pairs_list.append((seed_ref, seed_noise))
    seed_pairs = tuple(seed_pairs_list)
    if len(set(seed_pairs)) != len(seed_pairs):
        raise ValueError("manifest seed_pairs must be unique")
    for name in ("train", "validation"):
        value = document[name]
        if not isinstance(value, list) or not value:
            raise ValueError(f"manifest {name} must be a non-empty path list")
        if not all(isinstance(item, str) and item for item in value):
            raise TypeError(f"manifest {name} entries must be non-empty strings")
    base = source.resolve().parent
    train_paths = tuple((base / item).resolve() for item in document["train"])
    validation_paths = tuple(
        (base / item).resolve() for item in document["validation"]
    )
    if len(set(train_paths)) != len(train_paths):
        raise ValueError("manifest train contains duplicate cache paths")
    if len(set(validation_paths)) != len(validation_paths):
        raise ValueError("manifest validation contains duplicate cache paths")
    if set(train_paths) & set(validation_paths):
        raise ValueError("a cache path cannot appear in both train and validation")
    return PRICacheManifest(
        train=train_paths,
        validation=validation_paths,
        protocol=PRICacheProtocol(
            producer_version=producer_version,
            teacher_config_fingerprint=teacher_config_fingerprint,
            num_inference_steps=num_inference_steps,
            selected_steps=selected_steps,
            seed_pairs=seed_pairs,
        ),
    )


def validate_split_isolation(
    train_records: Sequence[PRITrajectoryRecord],
    validation_records: Sequence[PRITrajectoryRecord],
) -> None:
    """Reject source, sample, or exact sample/seed leakage across splits."""

    if not train_records or not validation_records:
        raise ValueError("train_records and validation_records must be non-empty")
    for record in train_records:
        if record.metadata.split != "train":
            raise ValueError("all train_records must declare split='train'")
    for record in validation_records:
        if record.metadata.split != "validation":
            raise ValueError("all validation_records must declare split='validation'")

    for split_name, records in (
        ("train", train_records),
        ("validation", validation_records),
    ):
        keys = [
            (
                record.metadata.sample_id,
                record.metadata.seed_ref,
                record.metadata.seed_noise,
            )
            for record in records
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"duplicate sample/seed trajectories within {split_name} split"
            )

    train_groups = {record.metadata.source_group_id for record in train_records}
    validation_groups = {
        record.metadata.source_group_id for record in validation_records
    }
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"source_group_id leakage across splits: {sorted(overlap)}")
    train_samples = {record.metadata.sample_id for record in train_records}
    validation_samples = {record.metadata.sample_id for record in validation_records}
    overlap = train_samples & validation_samples
    if overlap:
        raise ValueError(f"sample_id leakage across splits: {sorted(overlap)}")
    train_seed_keys = {
        (record.metadata.sample_id, record.metadata.seed_ref, record.metadata.seed_noise)
        for record in train_records
    }
    validation_seed_keys = {
        (record.metadata.sample_id, record.metadata.seed_ref, record.metadata.seed_noise)
        for record in validation_records
    }
    overlap_keys = train_seed_keys & validation_seed_keys
    if overlap_keys:
        raise ValueError("sample/seed trajectory leakage across splits")


def validate_manifest_protocol(
    manifest: PRICacheManifest,
    train_entries: Sequence[PRICacheIndexEntry | PRITrajectoryRecord],
    validation_entries: Sequence[PRICacheIndexEntry | PRITrajectoryRecord],
) -> None:
    """Validate global trajectory conventions and complete per-sample seeds."""

    if not isinstance(manifest, PRICacheManifest):
        raise TypeError("manifest must be a PRICacheManifest")
    validate_split_isolation(train_entries, validation_entries)
    protocol = manifest.protocol
    all_entries = tuple(train_entries) + tuple(validation_entries)
    for entry in all_entries:
        metadata = entry.metadata
        if metadata.producer_version != protocol.producer_version:
            raise ValueError("record producer_version does not match manifest protocol")
        if metadata.teacher_config_fingerprint != protocol.teacher_config_fingerprint:
            raise ValueError(
                "record teacher_config_fingerprint does not match manifest protocol"
            )
        if metadata.num_inference_steps != protocol.num_inference_steps:
            raise ValueError("record num_inference_steps does not match manifest protocol")
        if metadata.selected_steps != protocol.selected_steps:
            raise ValueError("record selected_steps does not match manifest protocol")

    expected_seeds = set(protocol.seed_pairs)
    for split_name, entries in (
        ("train", train_entries),
        ("validation", validation_entries),
    ):
        by_sample: dict[str, set[tuple[int, int]]] = {}
        for entry in entries:
            metadata = entry.metadata
            by_sample.setdefault(metadata.sample_id, set()).add(
                (metadata.seed_ref, metadata.seed_noise)
            )
        for sample_id, actual_seeds in by_sample.items():
            if actual_seeds != expected_seeds:
                raise ValueError(
                    f"{split_name} sample {sample_id!r} does not contain the complete "
                    "manifest seed_pairs set"
                )


__all__ = [
    "CACHE_METADATA_KEY",
    "PRI_CACHE_SCHEMA",
    "PRI_CACHE_VERSION",
    "PRICacheIndexEntry",
    "PRICacheManifest",
    "PRICacheMetadata",
    "PRICacheProtocol",
    "PRITrajectoryRecord",
    "index_pri_cache",
    "load_cache_manifest",
    "read_pri_cache",
    "validate_manifest_protocol",
    "validate_split_isolation",
    "write_pri_cache",
]
