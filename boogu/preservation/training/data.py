"""Strict local paired manifests; target data is used only for supervision."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..runtime import load_permissions, source_rgb_sha256


@dataclass(frozen=True)
class Pair:
    pair_id: str
    source: Path
    target: Path
    instruction: str
    permissions: Path
    supervision: Path
    split: str
    source_group: str
    admitted: bool
    same_frame: bool


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_pair(row):
    """Validate/load one pair; callers may discard it before loading the next."""
    source, target = read_rgb(row.source), read_rgb(row.target)
    if source.size != target.size or any(side % 16 for side in source.size):
        raise ValueError(
            f"{row.pair_id}: source/target canvas must match and be divisible by 16"
        )
    if row.same_frame is not True:
        raise ValueError(
            "same_frame must explicitly attest registered source/target pixels"
        )
    keep, modify, valid = load_permissions(row.permissions, source)
    shape = (source.height, source.width)
    with np.load(row.supervision, allow_pickle=False) as archive:
        names = set(archive.files)
        if not {"allowed_edit", "edit_core"} <= names:
            raise ValueError("supervision requires allowed_edit and edit_core")
        masks = {}
        for name in names:
            if name not in {"allowed_edit", "edit_core"} and not re.fullmatch(
                r"protected_roi__[A-Za-z0-9_-]+", name
            ):
                raise ValueError(f"unknown supervision field: {name}")
            value = archive[name]
            if value.dtype != np.bool_ or value.shape != shape:
                raise ValueError(f"{name} must be boolean and source-canvas aligned")
            masks[name] = value.copy()
    if np.any(masks["edit_core"] & ~masks["allowed_edit"]):
        raise ValueError("edit_core must be a subset of allowed_edit")
    return {
        "source": source,
        "target": target,
        "keep": keep,
        "modify": modify,
        "reference_valid": valid,
        "allowed_edit": masks["allowed_edit"],
        "edit_core": masks["edit_core"],
        "protected_roi_masks": {
            key.removeprefix("protected_roi__"): value
            for key, value in masks.items()
            if key.startswith("protected_roi__")
        },
    }


def pair_binding(row):
    """Bind metadata and every file to prevent stale feature-cache reuse."""
    metadata = asdict(row)
    files = {}
    for name in ("source", "target", "permissions", "supervision"):
        path = metadata[name]
        metadata[name] = str(path)
        files[name] = file_sha256(path)
    return {"record": metadata, "sha256": files}


def load_manifest(path):
    """Return validated train rows only; mixed/held-out manifests are rejected.

    `same_frame` and `admitted` are human/data-pipeline attestations, not an
    automatic image-registration or semantic-quality assessment. Split checks
    cover the supplied manifest; use stable source_group IDs across datasets.
    """
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        type(payload) is not dict
        or set(payload) != {"schema", "pairs"}
        or payload["schema"] != "boogu_preservation_pairs_v1"
        or type(payload["pairs"]) is not list
        or not payload["pairs"]
    ):
        raise ValueError("expected nonempty boogu_preservation_pairs_v1 manifest")
    rows, ids, contents, group_splits, image_splits = [], set(), set(), {}, {}
    fields = set(Pair.__dataclass_fields__)
    for record in payload["pairs"]:
        if type(record) is not dict or set(record) != fields:
            raise ValueError(f"each pair requires exactly {sorted(fields)}")
        for name in fields - {"admitted", "same_frame"}:
            if not isinstance(record[name], str) or not record[name].strip():
                raise ValueError(f"pair {name} must be a nonempty string")
        pair_id = record["pair_id"]
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", pair_id)
            or pair_id.endswith(".")
            or re.fullmatch(
                r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]",
                pair_id.split(".", 1)[0],
                re.IGNORECASE,
            )
        ):
            raise ValueError("pair_id must be a safe filename identifier")
        if pair_id.casefold() in ids:
            raise ValueError("duplicate pair_id (case-insensitive)")
        ids.add(pair_id.casefold())
        if record["split"] not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if record["admitted"] is not True or record["same_frame"] is not True:
            raise ValueError("every pair must be admitted and explicitly same_frame")
        values = dict(record)
        for name in ("source", "target", "permissions", "supervision"):
            values[name] = (path.parent / record[name]).resolve()
            if not values[name].is_file():
                raise ValueError(f"missing {name}: {values[name]}")
        row = Pair(**values)
        item = load_pair(row)
        source_hash, target_hash = (
            source_rgb_sha256(item[name]) for name in ("source", "target")
        )
        duplicate = (source_hash, target_hash, row.instruction)
        if duplicate in contents:
            raise ValueError("duplicate pair content")
        contents.add(duplicate)
        for key, table in (
            (row.source_group, group_splits),
            (source_hash, image_splits),
            (target_hash, image_splits),
        ):
            if key in table and table[key] != row.split:
                raise ValueError("source group or RGB image split leakage")
            table[key] = row.split
        rows.append(row)
        del item
    if any(row.split != "train" for row in rows):
        raise ValueError(
            "training requires a train-only manifest; validation/test rows are forbidden"
        )
    return rows
