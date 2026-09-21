"""Safe disk I/O for Primary--Response--Invariant region maps."""

from __future__ import annotations

import os
import warnings
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from math import prod
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from PIL import Image, ImageOps

from .regions import PRIRegionProbabilities

MAX_PRI_ELEMENTS = 67_108_864

_IMAGE_FORMAT_BY_SUFFIX = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
}
_SUPPORTED_SUFFIXES = frozenset((*_IMAGE_FORMAT_BY_SUFFIX, ".npz"))
_SUPPORTED_TORCH_DTYPES = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
_SUPPORTED_NUMPY_DTYPES = frozenset(
    (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64))
)
_MAX_NPY_HEADER_SIZE = 10_000


@dataclass(frozen=True)
class _ValidatedNpySnapshot:
    shape: tuple[int, ...]
    normalized_shape: tuple[int, ...]
    dtype: np.dtype[Any]
    header_size: int
    payload: bytes


def _validate_dtype(dtype: Any) -> torch.dtype:
    try:
        supported = dtype in _SUPPORTED_TORCH_DTYPES
    except TypeError as error:
        raise TypeError(
            "dtype must be torch.float16, torch.bfloat16, torch.float32, "
            "or torch.float64"
        ) from error
    if not supported:
        raise TypeError(
            "dtype must be torch.float16, torch.bfloat16, torch.float32, "
            "or torch.float64"
        )
    return dtype


def _input_path(path: Any) -> Path:
    if isinstance(path, bytes) or not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or os.PathLike object")
    normalized = Path(path)
    if not normalized.exists():
        raise FileNotFoundError(normalized)
    if not normalized.is_file():
        raise IsADirectoryError(normalized)
    suffix = normalized.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported region-map suffix {normalized.suffix!r}; expected one of "
            f"{sorted(_SUPPORTED_SUFFIXES)}"
        )
    return normalized


def _check_element_limit(element_count: int) -> None:
    if element_count > MAX_PRI_ELEMENTS:
        raise ValueError(
            f"region map contains {element_count} values, exceeding "
            f"MAX_PRI_ELEMENTS={MAX_PRI_ELEMENTS}"
        )


def _load_image_probabilities(
    path: Path,
    *,
    dtype: torch.dtype,
    tolerance: float,
) -> PRIRegionProbabilities:
    expected_format = _IMAGE_FORMAT_BY_SUFFIX[path.suffix.lower()]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                actual_format = image.format
                if actual_format != expected_format:
                    raise ValueError(
                        f"image format {actual_format!r} does not match suffix "
                        f"{path.suffix!r} ({expected_format})"
                    )
                if image.mode != "RGB":
                    raise ValueError(
                        f"PRI region-map images must use RGB mode, got {image.mode!r}"
                    )
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("region-map images must have positive dimensions")
                _check_element_limit(3 * width * height)
                transposed = ImageOps.exif_transpose(image)
                try:
                    transposed_width, transposed_height = transposed.size
                    if transposed_width <= 0 or transposed_height <= 0:
                        raise ValueError(
                            "region-map images must have positive dimensions"
                        )
                    _check_element_limit(3 * transposed_width * transposed_height)
                    transposed.load()
                    pixels = np.array(transposed, dtype=np.uint8, copy=True)
                finally:
                    if transposed is not image:
                        transposed.close()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError(
            f"image exceeds Pillow decompression-bomb safety limit: {path}"
        ) from error
    except (OSError, SyntaxError) as error:
        raise ValueError(f"could not decode region-map image {path}") from error

    values = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(torch.float64)
    denominator = values.sum(dim=1, keepdim=True)
    if (denominator == 0).any().item():
        count = int((denominator == 0).sum().item())
        raise ValueError(
            f"region-map image contains {count} all-black pixel(s) with no PRI role"
        )
    probabilities = values / denominator
    regions = PRIRegionProbabilities.from_probabilities(
        probabilities,
        tolerance=tolerance,
    )
    return regions.to(dtype=dtype)


def _validate_npz_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 3:
        if shape[0] != 3:
            raise ValueError(
                "NPZ probabilities must have shape [3, H, W] or [B, 3, H, W]"
            )
        normalized_shape = (1, *shape)
    elif len(shape) == 4 and shape[1] == 3:
        normalized_shape = shape
    else:
        raise ValueError("NPZ probabilities must have shape [3, H, W] or [B, 3, H, W]")
    if any(dimension <= 0 for dimension in normalized_shape):
        raise ValueError("NPZ probabilities must have positive dimensions")
    _check_element_limit(prod(normalized_shape))
    return normalized_shape


def _read_npy_header(stream: BinaryIO) -> tuple[tuple[int, ...], np.dtype[Any], int]:
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, array_dtype = np.lib.format.read_array_header_1_0(
                stream,
                max_header_size=_MAX_NPY_HEADER_SIZE,
            )
        elif version == (2, 0):
            shape, _, array_dtype = np.lib.format.read_array_header_2_0(
                stream,
                max_header_size=_MAX_NPY_HEADER_SIZE,
            )
        else:
            raise ValueError(f"unsupported NPY header version {version!r}")
        header_size = stream.tell()
    except (EOFError, OSError, ValueError) as error:
        raise ValueError("invalid probabilities.npy header in NPZ archive") from error
    return tuple(shape), np.dtype(array_dtype), header_size


def _validate_npy_metadata(
    shape: tuple[int, ...], array_dtype: np.dtype[Any], header_size: int, file_size: int
) -> tuple[int, ...]:
    if array_dtype not in _SUPPORTED_NUMPY_DTYPES:
        raise TypeError("NPZ probabilities dtype must be float16, float32, or float64")
    normalized_shape = _validate_npz_shape(shape)
    expected_size = header_size + prod(normalized_shape) * array_dtype.itemsize
    if file_size != expected_size:
        raise ValueError(
            "probabilities.npy member length does not exactly match its header and "
            "array shape"
        )
    return normalized_shape


def _preflight_npz(file_object: Any) -> _ValidatedNpySnapshot:
    try:
        with zipfile.ZipFile(file_object, "r") as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != "probabilities.npy":
                raise ValueError(
                    "NPZ archive must contain exactly one physical probabilities.npy "
                    "member and no duplicates"
                )
            member = members[0]
            maximum_member_size = (
                MAX_PRI_ELEMENTS * np.dtype(np.float64).itemsize
                + _MAX_NPY_HEADER_SIZE
                + 16
            )
            if member.file_size > maximum_member_size:
                raise ValueError(
                    "NPZ probabilities exceed MAX_PRI_ELEMENTS before materialization"
                )
            with archive.open(member, "r") as stream:
                shape, array_dtype, header_size = _read_npy_header(stream)
            normalized_shape = _validate_npy_metadata(
                shape, array_dtype, header_size, member.file_size
            )

            with archive.open(member, "r") as stream:
                payload = stream.read(member.file_size + 1)
            if len(payload) != member.file_size:
                raise ValueError(
                    "probabilities.npy member length changed while creating its "
                    "validated snapshot"
                )

            snapshot_stream = BytesIO(payload)
            snapshot_shape, snapshot_dtype, snapshot_header_size = _read_npy_header(
                snapshot_stream
            )
            snapshot_normalized_shape = _validate_npy_metadata(
                snapshot_shape,
                snapshot_dtype,
                snapshot_header_size,
                len(payload),
            )
            if (
                snapshot_shape != shape
                or snapshot_dtype != array_dtype
                or snapshot_header_size != header_size
                or snapshot_normalized_shape != normalized_shape
            ):
                raise ValueError(
                    "probabilities.npy metadata changed while creating its validated "
                    "snapshot"
                )
            return _ValidatedNpySnapshot(
                shape=snapshot_shape,
                normalized_shape=snapshot_normalized_shape,
                dtype=snapshot_dtype,
                header_size=snapshot_header_size,
                payload=payload,
            )
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError("invalid or damaged NPZ zip archive") from error


def _load_npz_probabilities(
    path: Path,
    *,
    dtype: torch.dtype,
    tolerance: float,
) -> PRIRegionProbabilities:
    try:
        with path.open("rb") as file_object:
            snapshot = _preflight_npz(file_object)
        try:
            probabilities = np.load(BytesIO(snapshot.payload), allow_pickle=False)
        except ValueError as error:
            raise ValueError(
                "NPZ probabilities must use a non-pickle floating dtype"
            ) from error
        if not isinstance(probabilities, np.ndarray):
            raise TypeError("validated probabilities.npy did not decode as an array")
        if (
            tuple(probabilities.shape) != snapshot.shape
            or probabilities.dtype != snapshot.dtype
        ):
            raise ValueError(
                "decoded probabilities.npy metadata differs from its validated snapshot"
            )
        if tuple(probabilities.shape) != snapshot.normalized_shape:
            probabilities = probabilities[None]
        promoted = np.array(probabilities, dtype=np.float64, copy=True)
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"could not decode NPZ region map {path}") from error

    validated = PRIRegionProbabilities.from_probabilities(
        torch.from_numpy(promoted),
        tolerance=tolerance,
    )
    if dtype == torch.float64:
        return validated
    return validated.to(dtype=dtype)


def load_pri_region_map(
    path: str | os.PathLike[str],
    *,
    dtype: torch.dtype = torch.float32,
    tolerance: float = 1e-4,
) -> PRIRegionProbabilities:
    """Load and validate one RGB image or NPZ PRI region map on CPU."""

    dtype = _validate_dtype(dtype)
    normalized_path = _input_path(path)
    if normalized_path.suffix.lower() == ".npz":
        return _load_npz_probabilities(
            normalized_path,
            dtype=dtype,
            tolerance=tolerance,
        )

    return _load_image_probabilities(
        normalized_path,
        dtype=dtype,
        tolerance=tolerance,
    )


def load_pri_region_map_list(
    paths: Sequence[str | os.PathLike[str]],
    *,
    dtype: torch.dtype = torch.float32,
    tolerance: float = 1e-4,
) -> list[PRIRegionProbabilities]:
    """Load one sample per path while preserving independently sized maps."""

    if isinstance(paths, (str, bytes, os.PathLike)) or not isinstance(paths, Sequence):
        raise TypeError("paths must be a non-empty sequence of path-like objects")
    if not paths:
        raise ValueError("paths must be a non-empty sequence")
    loaded = [
        load_pri_region_map(path, dtype=dtype, tolerance=tolerance) for path in paths
    ]
    if any(regions.values.shape[0] != 1 for regions in loaded):
        raise ValueError("each path in paths must contain exactly one sample")
    return loaded


def stack_pri_region_maps(
    regions: Sequence[PRIRegionProbabilities],
) -> PRIRegionProbabilities:
    """Stack explicitly compatible single-sample PRI maps into one batch."""

    if isinstance(regions, (str, bytes, os.PathLike)) or not isinstance(
        regions, Sequence
    ):
        raise TypeError("regions must be a non-empty sequence")
    if not regions:
        raise ValueError("regions must be a non-empty sequence")
    if any(not isinstance(item, PRIRegionProbabilities) for item in regions):
        raise TypeError("all regions must be PRIRegionProbabilities instances")
    first = regions[0]
    if any(item.values.shape[0] != 1 for item in regions):
        raise ValueError("each region map must contain one sample")
    if any(item.values.shape[-2:] != first.values.shape[-2:] for item in regions[1:]):
        raise ValueError("all region maps must have the same spatial dimensions")
    if any(item.values.dtype != first.values.dtype for item in regions[1:]):
        raise ValueError("all region maps must have the same dtype")
    if any(item.values.device != first.values.device for item in regions[1:]):
        raise ValueError("all region maps must be on the same device")
    if any(item.tolerance != first.tolerance for item in regions[1:]):
        raise ValueError("all region maps must use the same tolerance")
    return PRIRegionProbabilities.from_probabilities(
        torch.cat([item.values for item in regions], dim=0),
        tolerance=first.tolerance,
    )


load_causal_region_map = load_pri_region_map


__all__ = [
    "MAX_PRI_ELEMENTS",
    "load_causal_region_map",
    "load_pri_region_map",
    "load_pri_region_map_list",
    "stack_pri_region_maps",
]
